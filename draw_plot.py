#!/usr/bin/env python

import matplotlib.pyplot as plt

x = list(xrange(10, 110, 10))
# y = [115.582, 61.368, 50.884, 44.300, 46.626, 37.189, 36.253, 35.630, 34.832]
y = [0.791476, 0.811126, 0.826645, 0.831273, 0.839994, 0.851707, 0.858969, 0.881183, 0.878757, 0.884136]
plot1, = plt.plot(x, y, 'b-')

# plt.xlim(0, 100)

plt.title('Runtime vs Factor')
plt.xlabel('Factor')
plt.ylabel('Runtime')
# plt.legend([plot1, plot2, plot3], ['N = 10^4', 'N = 10^5', 'N = 10^7'])

plt.show()