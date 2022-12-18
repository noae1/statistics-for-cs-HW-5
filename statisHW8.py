import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import pearsonr
import pandas as pd
from scipy.stats import t
import seaborn as sb

# Q4

# a
beta = 1 - norm.cdf(norm.ppf(0.05) , loc = -2.5, scale =  1)
print(beta)

# b
count_reject_based_on_H0 = 0
count_not_reject_based_on_HA = 0
for i in range(10000):
    # H-0
    x_0 = norm.rvs(loc = 175, scale = 10, size=25)
    x_A = norm.rvs(loc=170, scale=10, size=25)

    mean_0 = np.mean(x_0)
    mean_A = np.mean(x_A)

    Z_0 = (mean_0 - 175) / 2
    Z_A = (mean_A - 175) / 2

    count_reject_based_on_H0 += (1 if (Z_0 <= norm.ppf(0.05)) else 0)
    count_not_reject_based_on_HA += (1 if (Z_A > norm.ppf(0.05)) else 0)


alfa_midgami = np.sum(count_reject_based_on_H0)/10000
beta_midgami = np.sum(count_not_reject_based_on_HA)/10000

print(alfa_midgami)
print(beta_midgami)

# c
print(t.ppf(0.05,df=24))

# d
print()
n = 10000

count_reject_based_on_H0 = 0
for i in range(n):
    x_0 = t.rvs(size=n , df = 24)
    count_reject_based_on_H0 += (1 if (x_0[i] <= t.ppf(0.05, df=24)) else 0)

alfa_midgami_unknown_sd = np.sum(count_reject_based_on_H0) / n
print(alfa_midgami_unknown_sd)


count_not_reject_based_on_HA = 0
for i in range(n):
    x_A = norm.rvs(loc=170, scale=10, size=25)
    hefresh = x_A - 170
    omed_sd = math.sqrt( np.sum( np.power(hefresh, 2) ) / 24 )
    T = (np.mean(x_A) - 175) / (omed_sd / math.sqrt(25))
    count_not_reject_based_on_HA += (1 if (T > t.ppf(0.05, df=24)) else 0)

beta_midgami_unknown_sd = np.sum(count_not_reject_based_on_HA) / n
print(beta_midgami_unknown_sd)


# e
# example :
for i in range(10000):
    x_0 = norm.rvs(loc=175, scale=10, size=25)

    # unknown sd:
    hefresh = x_0 - 175
    omed_sd = math.sqrt( np.sum( np.power(hefresh, 2) ) / 24 )
    T = (np.mean(x_0) - 175) / (omed_sd / math.sqrt(25))

    # known sd:
    Z = (np.mean(x_0) - 175) / 2

    if T < t.ppf(0.05, df=24) and Z > norm.ppf(0.05):
        print(T)
        print(Z)
        break
