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

# Q1

def q1_a (alfa , n):
    k = 10 ** 4
    length_k = []
    is_in_k = []   # 1 for True, 0 for False
    p_another_sample_is_in_k = []

    for i in range(k):
        # samples of normal distribution with loc (=mean) = 175, scale (=sd) = 10:
        samples = stats.norm.rvs(loc=175, scale=10, size=n)
        epsilon = norm.ppf(1 - (alfa/2)) * 10 / math.sqrt(n)
        mean = np.mean(samples)
        low = mean - epsilon
        high = mean + epsilon
        #print("confidence intervals n = ",n," alfa = ",alfa," :  [",low," , ",high," ]")
        #print(" length = ", 2 * epsilon)
        is_in = 1 if (low <= 175 and 175 <= high) else 0
        #print("Does the range contain 100 : ",is_in)
        another_sample_is_in = norm.cdf(high, loc=175, scale=10) - norm.cdf(low, loc=175, scale=10)
        #print("p another sample is in range : ",another_sample_is_in)

        length_k.append(2 * epsilon)
        is_in_k.append(is_in)
        p_another_sample_is_in_k.append(another_sample_is_in)

    print("alfa = ",alfa,",  n = ",n)
    print("avg length = ", np.mean(length_k))
    print("Standard deviation of the k lengths = ", np.std(length_k))
    sum = np.sum(is_in_k)
    print("probability that the range contain 100 : ", sum / k)
    print("probability that another sample is in rang : ",np.mean(p_another_sample_is_in_k))
    print()


# a
print(" ----- a ----- Based on the k samples :\n")
n_arr = [10, 20, 40, 80]
alfa = 0.05
for i in range(4):
    q1_a(alfa , n_arr[i])

# b
print(" ----- b ----- Based on the k samples :\n")
n = 30
alfa_arr = [0.05, 0.1, 0.2]
for i in range(3):
    q1_a(alfa_arr[i] , n)


# Q2
n = np.linspace(0,30,10000)
d = 3.92 / (n ** 0.5)
plt.plot(n, d,'r', label='d(n)')
plt.legend()
plt.show()

d_16 = 3.92 / (16 ** 0.5)
d_15 = 3.92 / (15 ** 0.5)
print(d_15)
print(d_16)


# Q3

# create Smooth histogram - taken from HW 2
def kernel_density(x, h):
    max_value = np.max(x)
    min_value = np.min(x)
    n = len(x)

    t = np.linspace(min_value-h, max_value+h, n)
    density = []

    for i in range(n):
        density.append( ( float (sum(x, h, t[i])) )/ (n * h) )

    plt.plot(t, density)

def cosine_kernel(u):
    if (abs(u) > 1/2):
        return 0
    return 1 + math.cos(2 * math.pi * u)

def sum(x,h,a):
    sum = 0
    for item in x:
        u = (item - a) / h
        sum += cosine_kernel(u)
    return sum


grades = [113, 105, 102, 104, 117, 123, 110, 108, 93, 96, 99, 107, 112, 82, 96]
kernel_density(grades, 30)
plt.ylabel("density")
plt.xlabel("grade")
plt.show()

mean = np.mean(grades)
epsilon = norm.ppf(1 - (0.05/2)) * 10 / math.sqrt(15)
print(epsilon)
print("low = ",(mean - epsilon))
print("up = ",(mean + epsilon))

# f
square_hefreshim = [(item - mean) ** 2 for item in grades]
omedan_var = np.sum(square_hefreshim) / 14
print(omedan_var)
print("omedan sd ",math.sqrt(omedan_var))

epsilon = 2.145 * 10.336 / math.sqrt(15)
print(epsilon)
print("low = ",(mean - epsilon))
print("up = ",(mean + epsilon))



# Q4

path = "C:/Users/Erezd/OneDrive/Desktop/keshet12.csv"
df = pd.read_csv(path)

# a
percent_of_votes = []
sum_vote = 0

for i in range(14):
    letter = chr (ord('A')+ i)
    voters_per_letter = df[letter].sum()
    sum_vote += voters_per_letter
    percent_of_votes.append(voters_per_letter)


percent_of_votes = percent_of_votes / sum_vote
print(percent_of_votes)

# b
L = []
U = []
alfa = 0.05
epsilon = norm.ppf(1 - (alfa/2)) * math.sqrt(0.25 / sum_vote)

for i in range(14):
    low = percent_of_votes[i] - epsilon
    high = percent_of_votes[i] + epsilon
    L.append(low)
    U.append(high)
    print(chr (ord('A')+ i)," - [",low," , ",high,"]")

# c
sorted_L = []
sorted_U = []
sorted_percent_of_votes = sorted(percent_of_votes)
sorted_miflagot = []

epsilon = norm.ppf(1 - (alfa/2)) * math.sqrt(0.25 / sum_vote)
for i in range(14):
    i = np.where( percent_of_votes == sorted_percent_of_votes[i] )[0][0]
    letter = chr(ord('A')+ i)
    sorted_miflagot.append(letter)

sorted_L = sorted_percent_of_votes - epsilon
sorted_U = sorted_percent_of_votes + epsilon
plt.scatter( sorted_miflagot , sorted_percent_of_votes)
plt.xlabel("miflagot")
plt.ylabel("percent of votes")
plt.vlines(x=sorted_miflagot , ymin= sorted_L, ymax= sorted_U, colors='black', ls='-', lw=2)
plt.show()

# d
voters_for_I = df['I'].sum()
voters_for_M = df['M'].sum()
sum = voters_for_I + voters_for_M
percent_of_votes_I = voters_for_I / sum  # p
percent_of_votes_M = voters_for_M / sum
print(percent_of_votes_I)
print(percent_of_votes_M)

print("new rs for p:")
alfa = 0.05    # changed to 0.01 for e
epsilon = norm.ppf(1 - (alfa/2)) * math.sqrt(0.25 / sum)
L = percent_of_votes_I - epsilon
U = percent_of_votes_I + epsilon
print("[",L," , ",U,"]")

print("new rs for d = 2p-1:")
print("[",2 * L - 1," , ",2 * U - 1,"]")
