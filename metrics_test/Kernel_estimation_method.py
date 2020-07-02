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


# # 2D metric implementation after [this paper](https://www.slac.stanford.edu/econf/C030908/papers/WEJT001.pdf)
# Here two datasets A and B are considered which may follow the same underlying distribution. These datasets are constructed by using the heteroscedastic distribution following the function  
# $7 \sin(x) + 3 |\cos(x/2)| \cdot \epsilon$ with $\epsilon\in\mathcal(0,1)$ ,
#
# which is computed several times and then turned into a two-dimensional histogram. This is done twice, sometimes changing the function slightly, depending on what distance measurement distribution we want to compute (same underlying distributions, different underlying distributions).

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


# To compare the two datasets, first the probability densities $f_A(x)$ and $f_B(x)$ are constructed using the 2D Kernel Estimation Method, which is already available in scikit. Later on we also need a 1D kernel estimation, which is done by the same function.

# +
def kernel_estimator1d(x):
    params = {'bandwidth': np.logspace(-1, 1, 20)} # the KDE has the bandwidth h as a parameter, which defines the "width of the kernel"
    grid = GridSearchCV(KernelDensity(), params) #to get the best h, a gridsearch is done
    grid.fit(x.reshape(-1, 1))                   #this gridsearch takes a rather long amount of time, so maybe h can be calculated beforehand
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


# -

# After the probability densities are found, we calculate the discriminant function defined as:
# $D(x) = \frac{f_A(x)}{f_A(x) + f_B(x)}$ .
# Similarly, D* is defined by replacing $f_B(x)$ with $f^*_A(x)$ which is obtained by generating n$_B$ random data points distributed according to $f_A(x)$. We also calculate $<D^*>$ in order to reduce the statistical fluctuations by computing D* several times and using the average.
# The cumulative distribution of D, called F, is then calculated, as well as F* and $<F^*>$. The Kolmogorov-Smirnov distance between F and $<F^*>$ is a measure of similarity between the datasets A and B, the KS distance between F* and $<F^*>$ is the expected distance if both sets came from the same distribution.

# +
def find_distribution(x, y_a, y_b, nbr): #Calculate D by drawing random samples from f_A and f_B and evaluating
    random_ints = np.random.randint(0, len(x), nbr)#D at those points - if A and B come from the same distribution
    y_a_sample = y_a[random_ints]                  #the peak of D should be at 0.5
    y_b_sample = y_b[random_ints]
    distribution = y_a_sample/(y_a_sample + y_b_sample)
    return distribution
    
def d_star(kde, length, Xgrid): #calculate D* by drawing samples from f_A and estimating the probability function
    new_samples = kde.sample(length) # of these new samples
    x_sample, y_sample = np.hsplit(new_samples, 2)
    x_sample = np.squeeze(x_sample)
    y_sample = np.squeeze(y_sample)
    log_dens, bandwidth, Xgrid, _ = kernel_estimator2d(x_sample, y_sample)
    return log_dens

def D_star_average(kde, length, Xgrid, nbr): #generate D* distribution nbr times, return all results
    average = []                             #average is done through building a histogram later
    for i in range(nbr):
        D_star = find_distribution(np.reshape(Xgrid_1, -1), log_dens_sample1, d_star(kde, length, Xgrid), 500)
        average.append(D_star)
    average = np.reshape(average, -1)
    return average

def make_D(kde, log_dens1, log_dens2, Xgrid, plot=False): #compute D, D* and <D*>, plot the histograms and the
    x_sample = np.arange(0, 1, 0.05)                     #calculated densities if plot=True, return the histograms
    D = find_distribution(np.reshape(Xgrid, -1), log_dens1, log_dens2, 500) #and the density values
    D_star = find_distribution(np.reshape(Xgrid, -1), log_dens1, f_star(kde, 1500, Xgrid), 500)
    D_aver = D_star_average(kde, 1500, Xgrid, 10)
    _, log_dens_D_aver = kernel_estimator1d(D_aver) #calculate the density functions of D, D* and <D*>
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

def make_F(log_dens_D, log_dens_D_star, log_dens_D_aver, plot=False): #compute F, F* and <F*> as the cumulative
    x_sample = np.arange(0, 1, 0.05)                                  #distribution function of D 
    F = np.cumsum(np.exp(log_dens_D))                                 #plot the functions if plot=True
    F_star = np.cumsum(np.exp(log_dens_D_star))                        
    F_aver = np.cumsum(np.exp(log_dens_D_aver))
    if plot:
        plt.plot(x_sample, F, 'r', label="F")
        plt.plot(x_sample, F_star, 'b', label="F*")
        plt.plot(x_sample, F_aver, 'g', label="<F*>")
        plt.legend()
        plt.show()
    ks_d = np.max(abs(F - F_aver)) #compute the distance measurements as a Kolmogorov-Smirnov distance
    ks_d_star = np.max(abs(F_star - F_aver))
    return F, F_star, F_aver, ks_d, ks_d_star



# -

repeats = 100
change1 = 7
change2 = 5
for j in range(5): #make difference between datasets higher
    ks_dist = []
    ks_real = []
    for i in range(repeats): #repeat calculations to get distance measure distribution
        x1, y1 = generate_norm_data(450, change1) #generate heteroscedastic dataset several times to be able to
        x2, y2 = generate_norm_data(450, change1) #compute a 2D histogram
        x3, y3 = generate_norm_data(450, j)
        x4, y4 = generate_norm_data(450, j)

        x_test1 = np.concatenate((x1, x2))
        y_test1 = np.concatenate((y1, y2))
        x_test2 = np.concatenate((x3, x4))
        y_test2 = np.concatenate((y3, y4))
    
        log_dens_sample1, bandwidth_1, Xgrid_1, kde1 = kernel_estimator2d(x_test1, y_test1) #estimate the PDF of
        log_dens_sample2, bandwidth_2, Xgrid_2, kde2 = kernel_estimator2d(x_test2, y_test2) #two distributions with KDE
        
        #calculate D, D*, <D*>, and F plus the distances between F and <F*> and F* and <F*>
        _, log_dens_D, _, log_dens_D_star, _, log_dens_D_aver = make_D(kde1, log_dens_sample1, log_dens_sample2, Xgrid_1, plot=False)
        _, _, _, ks_d, ks_d_star = make_F(log_dens_D, log_dens_D_star, log_dens_D_aver, plot = False)
        ks_dist.append(ks_d)
        ks_real.append(ks_d_star)
    df = pd.DataFrame({"F - <F*>": ks_dist, "F* - <F*>": ks_real})
    df.to_csv("./data/metrics/2Dmetrics_changedist_{}repeats_change{}_{}.csv".format(repeats, change1, j), index=False)
