import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import pearsonr
import pandas as pd

import seaborn as sb

# Q1

# ------------ uniform distribution ----------

# a
print("uniform distribution : ")
k = int (math.pow(10,4))
y1 = []  # first omed
y2 = []  # second omed

for i in range(k):
    # teta = high (parameter)
    x = np.random.uniform(0, 5, 10)
    y1.append( 2 * np.mean(x) )
    y2.append( np.max(x) )

# amptiri :
var1 = np.var(y1)
bias1 = 5 - np.mean(y1)
MSE1 = bias1 ** 2 + var1

var2 = np.var(y2)
bias2 = 5 - np.mean(y2)
MSE2 = bias2 ** 2 + var2

print ("MSE 1 = ",MSE1)
print("MSE 2 = ", MSE2)

print("theory calc for uniform distribution: ")
var_x = (5 ** 2) / 12  # theory :  (b-a)^2 / 12

# c
sb.kdeplot(y1, color= 'r', label='teta 1')
sb.kdeplot(y2, color= 'g', label='teta 2')
plt.xlabel("x")
plt.axvline(5, 0,max(y1))    # vertical line x=5
plt.legend()
plt.show()

# d
y =[]
random_20_from_y1 = random.sample(y1, 20)

print ("section d for uniform distribution:")
print ("var : ",np.var(random_20_from_y1))
print ("mean : ",np.mean(random_20_from_y1))

print()

# ------------ exp distribution ----------
# a
print ("exp distribution : ")
k = int (math.pow(10,4))
y1 = []  # first omed
y2 = []  # second omed

for i in range(k):
    x = np.random.exponential(scale=1, size=10)  # scale = std
    y1.append( 1 / np.mean(x) )
    y2.append( np.log(2) / np.median(x) )

# b
# amptiri :
var1 = np.var(y1)
bias1 = 1 - np.mean(y1)
MSE1 = bias1 ** 2 + var1

var2 = np.var(y2)
bias2 = 1 - np.mean(y2)
MSE2 = bias2 ** 2 + var2

print ("MSE 1 = ",MSE1)
print("MSE 2 = ", MSE2)

# c
sb.kdeplot(y1, color= 'r', label='gama 1')
sb.kdeplot(y2, color= 'g', label='gama 2')
plt.xlabel("x")
plt.axvline(1, 0,max(y1))    # vertical line x=5
plt.legend()
plt.show()

# d
random_20_from_y1 = random.sample(y1, 20)

print ("section d for exp distribution:")
print ("var : ",np.var(random_20_from_y1))
print ("mean : ",np.mean(random_20_from_y1))


# -------------------------------------------------------------------------





# Q4

gama = 0.1 #0.1 or 10

k = int (math.pow(10,4))
y_exp = []  # first omed
y_poi = []  # second omed

for i in range(k):
    x_exp = np.random.exponential(scale= 1/gama, size=10)  # scale = std
    x_poi = np.random.poisson( gama, size=10)
    y_exp.append( 1 / np.mean(x_exp) )
    y_poi.append( np.mean(x_poi) )

var_exp = np.var(y_exp)
bias_exp = gama - np.mean(y_exp)
MSE_exp = bias_exp ** 2 + var_exp

var_poi = np.var(y_poi)
bias_poi = gama - np.mean(y_poi)
MSE_poi = bias_poi ** 2 + var_poi

print ("MSE exp = ",MSE_exp)
print("MSE poi = ", MSE_poi)




