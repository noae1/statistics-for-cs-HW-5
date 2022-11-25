import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import pearsonr
import pandas as pd

import seaborn as sb

# Q2

# b
A = np.random.randint(1, 7, 100000)
B = np.random.randint(1, 7, 100000)
X2 = 0.5 * (A + B)
sb.kdeplot(X2)
plt.xlabel("X2")

# c
mean = np.mean(X2)
sd = np.std(X2)
print("mean = ", mean)
print("sd = ", sd)

# d
x = np.arange(0, 7, 0.001)
#plot normal distribution with mean 3.5 and standard deviation math.sqrt(35/24)
plt.plot(x, norm.pdf(x, 3.5, math.sqrt(35/24)), color = 'r')
plt.show()

# e
def X_n_vector (n):
    A = np.random.randint(1, 7, 100000)
    for i in range(n-1):
        B = np.random.randint(1, 7, 100000)
        A += B
    X_n = [ item/n for item in A]
    mean = np.mean(X_n)
    sd = np.std(X_n)
    return X_n, mean, sd


X6, mean6, sd6 = X_n_vector(6)
X8, mean8, sd8 = X_n_vector(8)
X10, mean10, sd10= X_n_vector(10)
X12, mean12, sd12 = X_n_vector(12)

# x4
sb.kdeplot(X6, color='g', label='X6')
plt.legend()
print ("X4: mean, sd = ",mean6," ,", sd6)
plt.show()

# x5
sb.kdeplot(X8, color='r', label='X8')
plt.legend()
print ("X8: mean, sd = ",mean8," ,", sd8)
plt.show()

# x6
sb.kdeplot(X10, color='b', label='X10')
plt.legend()
print ("X10: mean, sd = ",mean10," ,", sd10)
plt.show()

# x8
sb.kdeplot(X12, label='X12')
plt.legend()
print ("X12: mean, sd = ",mean12," ,", sd12)
# plot normal distribution :
# plt.plot(x, norm.pdf(x, 3.5, math.sqrt(35/144)), color = 'purple')  # X12
plt.show()

# f
q = norm.ppf(0.9, loc = 3.5, scale = sd12)    # x_0.95
print(q)



# Q4

# 100 linearly spaced numbers
x = np.linspace(50,70,100)

y1 = x**2 - 120*x +3600
plt.plot(x,y1, 'r', label = '1')

y2 = [2.5 for item in x]
plt.plot(x,y2, 'g', label = '2')

y3 = 1/36 * (x**2) - (10/3 * x) + 100 + 125/72
plt.plot(x,y3, 'b', label = '3')

plt.ylim(0,10)
plt.ylabel('MSE')
plt.xlabel('mu')
plt.legend()
plt.show()

# d
t = [81,95,85,75,98,100,85,86,92,91]
print (np.mean(t))  # = 88.8





