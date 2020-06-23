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
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from sklearn.model_selection import GridSearchCV
from ekp_style import set_ekp_style
set_ekp_style(set_sizes=True, set_background=True, set_colors=True)


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
    #y = y + abs(np.min(y))
    #y_norm = y/np.sum(y)
    return x, y


def kernel_estimator(x_test, y_test):
    dens_sample = np.vstack((x_test, y_test))
    xgrid = np.linspace(np.min(x_test), np.max(x_test), 30)
    ygrid = np.linspace(np.min(y_test), np.max(y_test), 30)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(dens_sample.T)
    kde = grid.best_estimator_
    log_dens = kde.score_samples(np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T)
    return log_dens, grid.best_estimator_.bandwidth, Xgrid, kde


# +
def find_distribution(x, y_a, y_b, nbr):
    random_ints = np.random.randint(0, len(x), nbr)
    y_a_sample = y_a[random_ints]
    y_b_sample = y_b[random_ints]
    distribution = y_a_sample/(y_a_sample + y_b_sample)
    return distribution
    
def f_star(kde, length, Xgrid):
    new_samples = kde.sample(length)
    x_sample, y_sample = np.hsplit(new_samples, 2)
    x_sample = np.squeeze(x_sample)
    y_sample = np.squeeze(y_sample)
    log_dens, bandwidth, Xgrid, _ = kernel_estimator(x_sample, y_sample)
    return log_dens

def D_star_average(kde, length, Xgrid, nbr):
    average = []
    for i in range(nbr):
        D_star = find_distribution(np.reshape(Xgrid_1, -1), log_dens_sample1, f_star(kde, length, Xgrid), 500)
        average.append(D_star)
    average = np.reshape(average, -1)
    return average


# +
x1, y1 = generate_norm_data(450, 3)
x2, y2 = generate_norm_data(450, 3)
x3, y3 = generate_norm_data(750, 3)
x4, y4 = generate_norm_data(750, 3)

x_test1 = np.concatenate((x1, x2))
y_test1 = np.concatenate((y1, y2))
x_test2 = np.concatenate((x3, x4))
y_test2 = np.concatenate((y3, y4))
# -

h1, xedges, yedges, image = plt.hist2d(x_test1, y_test1, bins = (30, 30), norm = matplotlib.colors.PowerNorm(0.3), density=True)
h2, xedges2, yedges2, image2 = plt.hist2d(x_test2, y_test2, bins = (30, 30), norm = matplotlib.colors.PowerNorm(0.3), density=True)
plt.colorbar()
plt.show()

log_dens_sample1, bandwidth_1, Xgrid_1, kde1 = kernel_estimator(x_test1, y_test1)
dens_1 = np.exp(log_dens_sample1)
log_dens_sample2, bandwidth_2, Xgrid_2, kde2 = kernel_estimator(x_test2, y_test2)
dens_2 = np.exp(log_dens_sample2)

plt.imshow(dens_1.reshape(Xgrid_1.shape),
           origin='lower', aspect='auto',
           extent=[np.min(x_test1), np.max(x_test1), np.min(y_test1), np.max(y_test1)],
           cmap='Blues')
cb = plt.colorbar()
cb.set_label("density")

plt.imshow(dens_2.reshape(Xgrid_2.shape),
           origin='lower', aspect='auto',
           extent=[np.min(x_test2), np.max(x_test2), np.min(y_test2), np.max(y_test2)],
           cmap='Blues')
cb = plt.colorbar()
cb.set_label("density")

log_dens_star= f_star(kde1, 700, Xgrid_1)
dens_star = np.exp(log_dens_star)
plt.imshow(dens_star.reshape(Xgrid_2.shape),
           origin='lower', aspect='auto',
           extent=[np.min(x_test2), np.max(x_test2), np.min(y_test2), np.max(y_test2)],
           cmap='Blues')
cb = plt.colorbar()
cb.set_label("density")

D = find_distribution(np.reshape(Xgrid_1, -1), log_dens_sample1, log_dens_sample2, 500)
plt.hist(D, bins=20)
plt.show()

D_star = find_distribution(np.reshape(Xgrid_1, -1), log_dens_sample1, f_star(kde1, 1500, Xgrid_1), 500)
plt.hist(D_star)
plt.show()

average = D_star_average(kde1, 1500, Xgrid_1, 20)
plt.hist(average, weights = np.full(len(average), 20))
plt.show()
