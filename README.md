# DSGD-spark
Implement [DSGD-MF algorithm][1] in Python using [Spark][2]. It can be used as a recommandation system.

## How to run
### Parameters

    spark-submit dsgd_mf.py <num_factors> <num_workers> <num_iterations> <beta_value> <lambda_value> <inputV_filepath> <outputW_filepath> <outputH_filepath>

The code needs following parameters
> * num_factors: factor for matrix factorization
> * num_workers: number of workers 
> * num_iterations: number of iterations
> * beta_value: value of beta, a recommanded value is 0.6
> * lambda_value: value of lambda, a recommanded value is 0.1
> * inputV_filepath: path to the input file
> * outputW_filepath: path to the output file that is used to store W matrix
> * outputH_filepath: path to the output file that is used to store H matrix

### Input format
The input file should follow the given format. Use movie recommandation system as an example.

    <user_id1>, <movie_id1>, <rank>
    ...
    ...

### Output format
The output file of W and H will contain the value of W and H matrix seperately, deliminated by comma.


[1]: https://people.mpi-inf.mpg.de/~rgemulla/publications/gemulla11dsgd.pdf
[2]:https://spark.apache.org/
