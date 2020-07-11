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
from ekp_style import set_ekp_style
set_ekp_style(set_sizes=True, set_background=True, set_colors=True)
import pandas as pd


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


def generate_norm_data(length, change):
    x = np.zeros(length)
    y = np.zeros(len(x))
    for i in range(length):
        if i < length/3:
            x[i] = 2/5*np.random.randn()-4 
        elif i >= length/3 and i < length*2/3:
            x[i] = 0.9*np.random.randn() 
        else:    
            x[i] = 2/5*np.random.randn()+4

    y = change*np.sin(x) + 3*abs(np.cos(x/2))*np.random.randn(length)
    y = y + abs(np.min(y))
    #y_norm = y/np.sum(y)
    return x, y


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

repeats = 100
change1 = 7
change2 = 5
df = pd.DataFrame(columns = ['Kolmogorov1', 'Kolmogorov2', 'Kolmogorov3', 'Kolmogorov4', 'Kolmogorov5', 'Kolmogorov6', 'Kolmogorov7'])
for j in range(1, 8): #make difference between datasets higher
    k_d = []
    for i in range(repeats): #repeat calculations to get distance measure distribution
        x1, y1 = generate_norm_data(450, change1) #generate heteroscedastic dataset several times to be able to
        x2, y2 = generate_norm_data(450, change1) #compute a 2D histogram
        x3, y3 = generate_norm_data(450, j)
        x4, y4 = generate_norm_data(450, j)

        x_test1 = np.concatenate((x1, x2))
        y_test1 = np.concatenate((y1, y2))
        x_test2 = np.concatenate((x3, x4))
        y_test2 = np.concatenate((y3, y4))
        h1, xedges, yedges, image = plt.hist2d(x_test1, y_test1, bins = (30, 30), norm = matplotlib.colors.PowerNorm(0.3))
        h2, xedges2, yedges2, image2 = plt.hist2d(x_test2, y_test2, bins = (30, 30), norm = matplotlib.colors.PowerNorm(0.3))
        k_d.append(kolmogorov_2d(h1, h2, xedges, yedges))
    df["Kolmogorov{}".format(j)] = k_d
df.to_csv("./data/metrics/2DKolmogorov_changedist_{}repeats_change{}_{}.csv".format(repeats, change1, j), index=False)

df.head()

# +
plt.figure(figsize=(10,5))
plt.hist(df["Kolmogorov1"], bins= 20, label = "1: {}".format(np.round(np.mean(np.asarray(df["Kolmogorov1"])),4)))
#plt.hist(df["Kolmogorov2"], bins= 20, label = "2: {}".format(np.round(np.mean(np.asarray(df["Kolmogorov2"])),4)))
plt.hist(df["Kolmogorov3"], bins= 20, label = "3: {}".format(np.round(np.mean(np.asarray(df["Kolmogorov3"])),4)))
#plt.hist(df["Kolmogorov4"], bins= 20, label = "4: {}".format(np.round(np.mean(np.asarray(df["Kolmogorov4"])),4)))
plt.hist(df["Kolmogorov5"], bins= 20, label = "5: {}".format(np.round(np.mean(np.asarray(df["Kolmogorov5"])),4)))
#plt.hist(df["Kolmogorov6"], bins= 20, label = "6: {}".format(np.round(np.mean(np.asarray(df["Kolmogorov6"])),4)))
#plt.hist(df["Kolmogorov7"], bins= 20, label = "7: {}".format(np.round(np.mean(np.asarray(df["Kolmogorov7"])),4)))
plt.title("2D Kolmogorov distance")
plt.legend()

plt.savefig('./plots/metrics/2D/2DKolmogorov_100repeats_change1_3_5.pdf')
plt.show()

# +
x1, y1 = generate_norm_data(450, change1) #generate heteroscedastic dataset several times to be able to
x2, y2 = generate_norm_data(450, change1) #compute a 2D histogram
x3, y3 = generate_norm_data(450, 3)
x4, y4 = generate_norm_data(450, 3)

x_test1 = np.concatenate((x1, x2))
y_test1 = np.concatenate((y1, y2))
x_test2 = np.concatenate((x3, x4))
y_test2 = np.concatenate((y3, y4))
# %matplotlib inline
h1, xedges, yedges, image = plt.hist2d(x_test1, y_test1, bins = (30, 30), norm = matplotlib.colors.PowerNorm(0.3))
plt.show()
h2, xedges2, yedges2, image2 = plt.hist2d(x_test2, y_test2, bins = (30, 30), norm = matplotlib.colors.PowerNorm(0.3))
plt.show()
#plt.savefig("./plots/metrics/2D_hist_heteroscedastic_diff.png")

# +
x3, y3 = generate_norm_data(450, 5)
x4, y4 = generate_norm_data(450, 5)
x_test3 = np.concatenate((x3, x4))
y_test3 = np.concatenate((y3, y4))

plt.figure(figsize=(10,7))
plt.plot(x_test1, y_test1, '.b', label = "f(x) = 7sin(x) + ..")
plt.plot(x_test3, y_test3, '.g', label = "f(x) = 5sin(x) + ..")
plt.plot(x_test2, y_test2, '.r', label = 'f(x) = 3sin(x) + ..')
plt.title("Real data")
plt.legend(loc='upper right')
plt.savefig("./plots/metrics/realdata_7sin_5sin_3sin.pdf")
