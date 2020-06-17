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


def generate_data(length, diff):
    x = np.zeros(length)
    y = np.zeros(len(x))
    for i in range(length):
        if i < length/3:
            x[i] = 2/5*np.random.randn()-4 
        elif i >= length/3 and i < length*2/3:
            x[i] = 0.9*np.random.randn() 
        else:    
            x[i] = 2/5*np.random.randn()+4

    y = 7*np.sin(x) + diff*abs(np.cos(x/2))*np.random.randn(length)
    return x, y


def load_data(nr):
    x, y = np.loadtxt("./data/paper_heteroscedastic_data_test_{}.txt".format(nr))
    return x,y


def normalize_it(y):
    y = y + abs(np.min(y))
    y_norm = y/np.sum(y)
    return y_norm


def kolmogorov_2d(h1, h2, xedges, yedges):
    diff_point = 0
    diff_abs = 0
    for i in range(len(xedges)):
        for j in range(len(yedges)):
            diff_point = max(np.sum(h2[:i][:j]), np.sum(h1[i:][:j]), np.sum(h1[:i][j:]), np.sum(h1[i:][j:]))-max(np.sum(h2[:i][:j]), np.sum(h2[i:][:j]), np.sum(h2[:i][j:]), np.sum(h2[i:][j:]))
            diff_point = abs(diff_point)
            if diff_point > diff_abs:
                diff_abs = diff_point
    return diff_abs


x1, y1 = generate_data(750, 3)
y1 = normalize_it(y1)
x2, y2 = generate_data(750, 3)
y2 = normalize_it(y2)
x3, y3 = generate_data(750, 7)
y3 = normalize_it(y3)
x4, y4 = generate_data(750, 7)
y4 = normalize_it(y4)

x_test1 = np.concatenate((x1, x2))
y_test1 = np.concatenate((y1, y2))
x_test2 = np.concatenate((x3, x4))
y_test2 = np.concatenate((y3, y4))

# %matplotlib inline
h1, xedges, yedges, image = plt.hist2d(x_test1, y_test1, bins = (30, 30), norm = matplotlib.colors.PowerNorm(0.3))
plt.show()
h2, xedges2, yedges2, image2 = plt.hist2d(x_test2, y_test2, bins = (30, 30), norm = matplotlib.colors.PowerNorm(0.3))
plt.show()
plt.savefig("./plots/metrics/2D_hist_heteroscedastic.png")

print(kolmogorov_2d(h1, h2, xedges, yedges))
