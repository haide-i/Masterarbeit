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


def density_measurement(x):
    density = np.histogram(x, bins='auto')
    return density


def load_data(nr):
    x, y = np.loadtxt("./data/paper_heteroscedastic_data_test_{}.txt".format(nr))
    return x,y


def kullback_leibler(x1, x2):
    D_KL = np.sum(x1 * np.log(x1/x2))
    return D_KL


x_test1, y_test1 = load_data(0)
x_test2, y_test2 = load_data(1)
x_test3, y_test3 = load_data(2)
x_test4, y_test4 = load_data(3)

plt.plot(x_test1, y_test1, '.b')

plt.plot(x_test2, y_test2, '.r')

x_hist1, _, _ = plt.hist(x_test1, bins = 'auto', density=True)
x_hist2, _, _ = plt.hist(x_test2, bins = 'auto', density=True)
x_hist3, _, _ = plt.hist(x_test3, bins = 'auto', density=True)
x_hist4, _, _ = plt.hist(x_test4, bins = 'auto', density=True)

# +
D_KL12 = kullback_leibler(x_hist1, x_hist2)
D_KL23 = kullback_leibler(x_hist2, x_hist3)
D_KL34 = kullback_leibler(x_hist3, x_hist4)
D_KL13 = kullback_leibler(x_hist1, x_hist3)
D_KL14 = kullback_leibler(x_hist1, x_hist4)
D_KL24= kullback_leibler(x_hist2, x_hist4)


print(D_KL12)
print(D_KL23)
print(D_KL34)
print(D_KL13)
print(D_KL14)
print(D_KL24)
