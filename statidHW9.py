from typing import Any

import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import gamma
from scipy.stats import pearsonr
import pandas as pd
from scipy.stats import t
import seaborn as sb

# Q1

# b
print(binom.ppf(q=0.05, n=40, p=0.5))
# answer = 15
print(binom.cdf(15, n=40, p=0.5))
print(binom.cdf(14, n=40, p=0.5))


# c
pi = binom.cdf(14, n=40, p=0.3)
print (pi)

# d
d = norm.ppf(q=0.05, loc=20, scale=math.sqrt(10))
print(d)

# e
pi = norm.cdf(14.798, loc=12, scale=math.sqrt(8.4))
print (pi)


# Q2
# b
print(t.ppf(0.95,df=9))

# c
x = [64, 75, 74, 78, 77, 66, 71, 79, 72, 67]
T = (np.mean(x) - 70) / (math.sqrt(np.var(x)/10))
print("T = ",T)
print(np.mean(x))

i = 0.01
alfa = 0.05
while(i <= 2):
    alfa += i
    low = t.ppf(1-alfa, df=9)
    if (T >= low):
        print("alfa = ",alfa)
        print("T >= low = ",low)
        break

# e
print(1 - norm.cdf(1.81))

# f
beta = norm.cdf(72.07, loc=72, scale=math.sqrt(1.6))
print(beta)


# Q3

# a
print(poisson.ppf(0.95,6))
# 10
print(1-poisson.cdf(9,6))  #10
print(1-poisson.cdf(10,6)) #11

# d
print (gamma.ppf(0.05, 30,0.2))

# f
