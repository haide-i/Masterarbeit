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


def generate_norm_data(length, diff):
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
    y = y + abs(np.min(y))
    y_norm = y/np.sum(y)
    return x, y_norm


def kernel_estimator(x, y):
    


# +
x1, y1 = generate_data(750, 3)
x2, y2 = generate_data(750, 3)
x3, y3 = generate_data(750, 7)
x4, y4 = generate_data(750, 7)

x_test1 = np.concatenate((x1, x2))
y_test1 = np.concatenate((y1, y2))
x_test2 = np.concatenate((x3, x4))
y_test2 = np.concatenate((y3, y4))
# -

h1, xedges1, yedges1 = np.histogram2d(x_test1, y_test1, bins = 20)
h2, xedges2, yedges2 = np.histogram2d(x_test2, y_test2, bins = 20)


