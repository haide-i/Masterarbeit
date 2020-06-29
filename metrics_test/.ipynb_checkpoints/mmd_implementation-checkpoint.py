# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy.random as rnd
import scipy.spatial as spt


def generate_rand_data(length, diff, change): #generate data with random distributed x-values and heteroscedastic y-values
    x = np.linspace(-5, 5, 1000)
    random_ints = np.random.randint(0, len(x), length)
    x = x[random_ints]
    y = change*np.sin(x) + diff*abs(np.cos(x/2))*np.random.randn(len(x))
    y = y + abs(np.min(y))
    y = y/np.sum(y)
    return x, y


def gaussfunction(x, y):#, sigma):
    return np.exp(-1/(2.*1**2)*abs(x-y)**2)


def maximum_mean_disc(x, y): #implement maximum mean discrepancy
    mmd = 0
    mixture = 0
    for i in range(len(x)):
        for j in range(len(y)):
            if i != j:
                mmd += gaussfunction(x[i], x[j]) + gaussfunction(y[i], y[j])
            mixture += gaussfunction(x[i], y[j]) + gaussfunction(x[j], y[i])
    mmd = 1/(len(x)*(len(x)-1))*mmd - 1/len(x)**2 * mixture
    return mmd


# +
mmd = []
for i in range(50):
    x1, y1 = generate_rand_data(500, 3, 7)
    x2, y2 = generate_rand_data(500, 3, 7)
    mmd.append(maximum_mean_disc(y1, y2))
    
plt.hist(mmd)

# +
mmd = []
for i in range(50):
    x1, y1 = generate_rand_data(500, 3, 2)
    x2, y2 = generate_rand_data(500, 3, 7)
    mmd.append(maximum_mean_disc(y1, y2))
    
plt.hist(mmd)
