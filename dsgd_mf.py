#!/usr/bin/env python

import sys
import numpy as np
from scipy import sparse

from pyspark import SparkContext, SparkConf

num_factors = int(sys.argv[1])
num_workers = int(sys.argv[2])
num_iterations = int(sys.argv[3])
beta_value = float(sys.argv[4])
lambda_value = float(sys.argv[5])
inputV_filepath = sys.argv[6]
outputW_filepath = sys.argv[7]
outputH_filepath = sys.argv[8]


def CreateHW() :
	with open(inputV_filepath, 'r') as f_in :	
		max_user = 0
		max_movie = 0
		for line in f_in :
			line = line.rstrip()
			eachline = line.split(",")
			for i in xrange(3) :
				eachline[i] = int(eachline[i])
			if max_user < eachline[0] :
				max_user = eachline[0]
			if max_movie < eachline[1] :
				max_movie = eachline[1]

	W = np.matrix(np.random.rand(max_user, max_movie))
	H = np.matrix(np.random.rand(max_user, max_movie))
	return W, H

def CreateMatrix(num_users, num_movies) :
	with open(inputV_filepath, 'r') as f_in :
		V = sparse.lil_matrix((num_users, num_movies))
		for line in f_in :
			line = line.rstrip()
			eachline = line.split(",")
			for i in xrange(3) :
				eachline[i] = int(eachline[i])
			V[eachline[0] - 1, eachline[1] - 1] = eachline[2]
	return V

def CalcGradient(block_tuple) :
	V_block = block_tuple[0]
	W_block = block_tuple[1]
	H_block = block_tuple[2]
	c = 0
	rows, cols = V_block.nonzero()
	for i, j in zip(rows, cols) :
		tmp = V_block[i,j] - (W_block[i, :] * H_block[:, j])[0, 0]
		eta = pow(100 + clk + c, -beta_value)
		new_W_block = W_block[i, :] - eta * \
					(-2 * tmp * H_block[:, j].transpose() \
					+ 2 * lambda_value * W_block[i, :] / V_block.tocsr()[i, :].nnz)
		H_block[:, j] = H_block[:, j] - eta * \
					(-2 * tmp * W_block[i, :].transpose() \
					+ 2 * lambda_value * H_block[:, j] / V_block.tocsc()[:, j].nnz)
		W_block[i, :] = new_W_block.copy()
		c += 1
	clock.add(len(rows))
	return (W_block, H_block)

def GetRowCol(strata) :
	rows = [i for i in xrange(strata[0], num_users, num_workers)]
	cols = [j for j in xrange(strata[1], num_movies, num_workers)]
	return (rows, cols)

def NextStrata(strata) :
	strata[1] = (strata[1] + 1) % num_workers
	return strata

def CalceError(V, W, H):
	error = 0.0
	V_err= W * H
	rows, cols = V.nonzero()
	for i, j in zip(rows,cols):
		tmp = V[i,j] - V_err[i,j]
		error += tmp * tmp
	error /= len(rows)
	return error

if __name__ == '__main__':
	#Create W H V
	W, H = CreateHW()
	num_users = W.shape[0]
	num_movies = H.shape[1]
	V = CreateMatrix(num_users, num_movies)

	# Initialize sc
	conf = SparkConf().setAppName('DSGD').setMaster('local[%d]' % num_workers)
	sc = SparkContext(conf=conf)

	# Intialize strata
	init_strata = [[i, v] for i, v in enumerate(np.random.permutation(num_workers))]
	S = sc.parallelize(init_strata)

	# Initialize clock
	clock = sc.accumulator(0)

	# Iteration
	for i in xrange(num_iterations) :
		# Get rows, cols from strata		
		split = S.map(GetRowCol).collect()
		# Get block from rows, cols
		matrices = []
		for row, col in split :
			V_block = V.tocsr()[row, :].tocsc()[:, col]
			W_block = W[row, :].copy()
			H_block = H[:, col].copy()
			matrices.append((V_block, W_block, H_block))
		# Set clock
		clk = clock.value
		# Calculate gradient
		matrices = sc.parallelize(matrices).map(CalcGradient).collect()
		# Upgrade W H
		for (row, col), (new_W, new_H) in zip(split, matrices) :
			W[row, :] = new_W
			H[:, col] = new_H
		# Get next strata
		S = S.map(NextStrata)

	# Save W H
	np.savetxt(outputW_filepath, W, delimiter=',')
	np.savetxt(outputH_filepath, H, delimiter=',')

	# Calculate reconstruction error
	print 'MSE: %f\n' % CalceError(V, W, H)

