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
import pandas as pd
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


# +
def kernel_estimator1d(x):
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(x.reshape(-1, 1))
    kde = grid.best_estimator_
    x_sample = np.arange(0, 1, 0.05)
    log_dens = kde.score_samples(x_sample.reshape(-1, 1))
    return x_sample, log_dens
    
    
def kernel_estimator2d(x_test, y_test):
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
    log_dens, bandwidth, Xgrid, _ = kernel_estimator2d(x_sample, y_sample)
    return log_dens

def D_star_average(kde, length, Xgrid, nbr):
    average = []
    for i in range(nbr):
        D_star = find_distribution(np.reshape(Xgrid_1, -1), log_dens_sample1, f_star(kde, length, Xgrid), 500)
        average.append(D_star)
    average = np.reshape(average, -1)
    return average

def make_D(kde, log_dens1, log_dens2, Xgrid, plot=False):
    x_sample = np.arange(0, 1, 0.05)
    D = find_distribution(np.reshape(Xgrid, -1), log_dens_sample1, log_dens_sample2, 500)
    D_star = find_distribution(np.reshape(Xgrid, -1), log_dens_sample1, f_star(kde1, 1500, Xgrid), 500)
    D_aver = D_star_average(kde1, 1500, Xgrid, 10)
    _, log_dens_D_aver = kernel_estimator1d(D_aver)
    _, log_dens_D = kernel_estimator1d(D)
    _, log_dens_D_star = kernel_estimator1d(D_star)    
    if plot:
        plt.hist(D, bins=20, histtype = 'step', color = 'red', label = 'D', density=True)
        plt.hist(D_star, bins=20, histtype = 'step', color = 'blue', label = 'D*', density=True)
        plt.hist(D_aver, bins=20, histtype = 'step', color = 'green', label = '<D*>', density = True)
        plt.plot(x_sample, np.exp(log_dens_D), 'r')
        plt.plot(x_sample, np.exp(log_dens_D_star), 'b')
        plt.plot(x_sample, np.exp(log_dens_D_aver), 'g')
        plt.legend()
        plt.show()
    return D, log_dens_D, D_star, log_dens_D_star, D_aver, log_dens_D_aver

def make_F(log_dens_D, log_dens_D_star, log_dens_D_aver, plot=False):
    x_sample = np.arange(0, 1, 0.05)
    F = np.cumsum(np.exp(log_dens_D))
    F_star = np.cumsum(np.exp(log_dens_D_star))
    F_aver = np.cumsum(np.exp(log_dens_D_aver))
    if plot:
        plt.plot(x_sample, F, 'r', label="F")
        plt.plot(x_sample, F_star, 'b', label="F*")
        plt.plot(x_sample, F_aver, 'g', label="<F*>")
        plt.legend()
        plt.show()
    ks_d = np.max(abs(F - F_aver))
    ks_d_star = np.max(abs(F_star - F_aver))
    return F, F_star, F_aver, ks_d, ks_d_star



# -

ks_dist = []
ks_real = []
repeats = 100
diff = 3
for i in range(repeats):
    x1, y1 = generate_norm_data(450, diff)
    x2, y2 = generate_norm_data(450, diff)
    x3, y3 = generate_norm_data(450, diff)
    x4, y4 = generate_norm_data(450, diff)

    x_test1 = np.concatenate((x1, x2))
    y_test1 = np.concatenate((y1, y2))
    x_test2 = np.concatenate((x3, x4))
    y_test2 = np.concatenate((y3, y4))
    
    log_dens_sample1, bandwidth_1, Xgrid_1, kde1 = kernel_estimator2d(x_test1, y_test1)
    log_dens_sample2, bandwidth_2, Xgrid_2, kde2 = kernel_estimator2d(x_test2, y_test2)

    _, log_dens_D, _, log_dens_D_star, _, log_dens_D_aver = make_D(kde1, log_dens_sample1, log_dens_sample2, Xgrid_1, plot=False)
    _, _, _, ks_d, ks_d_star = make_F(log_dens_D, log_dens_D_star, log_dens_D_aver, plot = False)
    ks_dist.append(ks_d)
    ks_real.append(ks_d_star)
    print(i)
df = pd.DataFrame({"F - <F*>": ks_dist, "F* - <F*>": ks_real})
df.to_csv("./data/metrics/2Dmetrics_samedist_{}repeats_diff{}.csv".format(repeats, diff), index=False)
plt.hist(ks_dist, bins = 20)
plt.show()

print(ks_dist)
print(ks_real)
ks_dist = np.array(ks_dist)
df = pd.DataFrame({"F - <F*>": ks_dist, "F* - <F*>": ks_real})
df.to_csv("./data/metrics/2Dmetrics_samedist_{}repeats_diff{}.csv".format(repeats, diff), index=False)
plt.hist(ks_dist, bins = 20)
plt.show()

# +
x1, y1 = generate_norm_data(450, 3)
x2, y2 = generate_norm_data(450, 3)
x3, y3 = generate_norm_data(750, 7)
x4, y4 = generate_norm_data(750, 7)

x_test1 = np.concatenate((x1, x2))
y_test1 = np.concatenate((y1, y2))
x_test2 = np.concatenate((x3, x4))
y_test2 = np.concatenate((y3, y4))
# -

h1, xedges, yedges, image = plt.hist2d(x_test1, y_test1, bins = (30, 30), norm = matplotlib.colors.PowerNorm(0.3), density=True, cmap='Blues')
h2, xedges2, yedges2, image2 = plt.hist2d(x_test2, y_test2, bins = (30, 30), norm = matplotlib.colors.PowerNorm(0.3), density=True, cmap='Blues')
plt.colorbar()
plt.show()

log_dens_sample1, bandwidth_1, Xgrid_1, kde1 = kernel_estimator2d(x_test1, y_test1)
dens_1 = np.exp(log_dens_sample1)
log_dens_sample2, bandwidth_2, Xgrid_2, kde2 = kernel_estimator2d(x_test2, y_test2)
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

D, log_dens_D, D_star, log_dens_D_star, D_aver, log_dens_D_aver = make_D(kde1, log_dens_sample1, log_dens_sample2, Xgrid_1, plot=True)

F, F_star, F_aver, ks_d, ks_d_star = make_F(log_dens_D, log_dens_D_star, log_dens_D_aver, plot = True)
print("K_S-distance of F and <F*>:", ks_d)
print("K_S-distance of F* and <F*>:", ks_d_star)
