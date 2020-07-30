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

import h5py as h5
import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
from glob import glob
from scipy.stats import wasserstein_distance
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import random
import ndks2d
home = os.getenv("HOME")
filenames = glob(home + '/data/photons/without_non_det/clean_photons_100x1E5_randpos_randdir_mod5_*.h5')
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


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

def D_star_average(kde, length, Xgrid, nbr, log_dens_sample1): #generate D* distribution nbr times, return all results
    average = []                             #average is done through building a histogram later
    for i in range(nbr):
        D_star = find_distribution(np.reshape(Xgrid, -1), log_dens_sample1, d_star(kde, length, Xgrid), 500)
        average.append(D_star)
    average = np.reshape(average, -1)
    return average

def make_D(kde, log_dens1, log_dens2, Xgrid, plot=False): #compute D, D* and <D*>, plot the histograms and the
    x_sample = np.arange(0, 1, 0.05)                     #calculated densities if plot=True, return the histograms
    D = find_distribution(np.reshape(Xgrid, -1), log_dens1, log_dens2, 500) #and the density values
    D_star = find_distribution(np.reshape(Xgrid, -1), log_dens1, d_star(kde, 1500, Xgrid), 500)
    D_aver = D_star_average(kde, 1500, Xgrid, 10, log_dens1)
    _, log_dens_D_aver = kernel_estimator1d(D_aver) #calculate the density functions of D, D* and <D*>
    _, log_dens_D = kernel_estimator1d(D)
    _, log_dens_D_star = kernel_estimator1d(D_star)    
    return D, log_dens_D, D_star, log_dens_D_star, D_aver, log_dens_D_aver

def make_F(log_dens_D, log_dens_D_star, log_dens_D_aver, plot=False): #compute F, F* and <F*> as the cumulative
    x_sample = np.arange(0, 1, 0.05)                                  #distribution function of D 
    F = np.cumsum(np.exp(log_dens_D))                                 #plot the functions if plot=True
    F_star = np.cumsum(np.exp(log_dens_D_star))                        
    F_aver = np.cumsum(np.exp(log_dens_D_aver))
    ks_d = np.max(abs(F - F_aver)) #compute the distance measurements as a Kolmogorov-Smirnov distance
    ks_d_star = np.max(abs(F_star - F_aver))
    return F, F_star, F_aver, ks_d, ks_d_star    

def kernel_test(x1, y1, x2, y2):
    log_dens_sample1, bandwidth_1, Xgrid_1, kde1 = kernel_estimator2d(x1, y1) #estimate the PDF of
    log_dens_sample2, bandwidth_2, Xgrid_2, kde2 = kernel_estimator2d(x2, y2) #two distributions with KDE
    _, log_dens_D, _, log_dens_D_star, _, log_dens_D_aver = make_D(kde1, log_dens_sample1, log_dens_sample2, Xgrid_1, plot=False)
    _, _, _, ks_d, ks_d_star = make_F(log_dens_D, log_dens_D_star, log_dens_D_aver, plot = False)
    return ks_d


# -

def kolmogorov_2d(x1, y1, x2, y2, bin1, bin2):
    h1, xedges, yedges = np.histogram2d(x1, y1, bins = (bin1, bin2))
    h2, xedges, yedges = np.histogram2d(x2, y2, bins = (bin1, bin2))
    diff_point = 0
    diff_abs = 0
    for i in range(len(xedges)):
        for j in range(len(yedges)):
            diff_point = max((np.sum(h1[:i][:j]) - np.sum(h2[:i][:j])), (np.sum(h1[i:][:j]) - np.sum(h2[i:][:j])),
                             (np.sum(h1[:i][j:]) - np.sum(h2[:i][j:])), (np.sum(h1[i:][j:]) - np.sum(h2[i:][j:])))
            diff_point = abs(diff_point)
            if diff_point > diff_abs:
                diff_abs = diff_point
    return diff_abs


dataframes = [pd.read_hdf(f) for f in filenames]
event_idx = []
for i in range(len(dataframes)):
    event_idx.append(np.unique(dataframes[i].evt_idx))

for i in range(len(dataframes)):
    for event in event_idx[i]:
        if event == 877:
            chosen_event = dataframes[i][dataframes[i].evt_idx == 877]
chosen_event.head(-1)

nr_of_photons = 100
use_rand = False
cls = ndks2d.ndKS()
alternative = True
for photons in (100, 1000):
    for i in range(len(dataframes)):
        if i > 0:
            del df
        df = pd.DataFrame(columns = ['evt_idx', 'KS_xy', 'KS_xt', 'KS_yt'])#, 'Kernel_xy', 'Kernel_xt', 'Kernel_yt'])
        KS_xy = []
        KS_xt = []
        KS_yt = []
        #Kernel_xy = []
        #Kernel_xt = []
        #Kernel_yt = []
        evt_idx = []
        for event in event_idx[i]:
            print(event)
            if len(dataframes[i][dataframes[i].evt_idx == event].index) > 10000:
                chosen_event = dataframes[i][dataframes[i].evt_idx == event]
                if (chosen_event.detection_pixel_x.max(axis = 0) - chosen_event.detection_pixel_x.min(axis = 0) != 0) \
                & (chosen_event.detection_pixel_y.max(axis = 0) - chosen_event.detection_pixel_y.min(axis = 0) != 0) \
                & (chosen_event.detection_time.max(axis = 0) - chosen_event.detection_time.min(axis = 0) != 0):
                    draw_rand = np.arange(0, len(chosen_event))
                    if use_rand:
                        rand_event = dataframes[i][dataframes[i].evt_idx == random.choice(event_idx[i])]
                        rand_length = np.arange(0, len(rand_event))
                    start = time.time()
                    for j in range(5):
                        event_sample1 = np.random.choice(draw_rand, photons)
                        x1 = torch.from_numpy(np.asarray(chosen_event.iloc[event_sample1,:].detection_pixel_x.to_numpy() + 23)/46.)
                        y1 = torch.from_numpy((chosen_event.iloc[event_sample1,:].detection_pixel_y.to_numpy() + 4.2)/(4.2+1.01))
                        t1 = torch.from_numpy((chosen_event.iloc[event_sample1,:].detection_time.to_numpy())/(chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0)))
                        if use_rand:
                                event_sample2 = np.random.choice(rand_length, photons)
                                x2 = torch.from_numpy((rand_event.iloc[event_sample2,:].detection_pixel_x.to_numpy() + 23)/46.)
                                y2 = torch.from_numpy((rand_event.iloc[event_sample2,:].detection_pixel_y.to_numpy() + 4.2)/(4.2+1.01))
                                t2 = torch.from_numpy(rand_event.iloc[event_sample2,:].detection_time.to_numpy()/(chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0)))
                        else:
                            event_sample2 = np.random.choice(draw_rand, photons)
                            x2 = torch.from_numpy((chosen_event.iloc[event_sample2,:].detection_pixel_x.to_numpy() + 23)/46.)
                            y2 = torch.from_numpy((chosen_event.iloc[event_sample2,:].detection_pixel_y.to_numpy() + 4.2)/(4.2+1.01))
                            t2 = torch.from_numpy(chosen_event.iloc[event_sample2,:].detection_time.to_numpy()/(chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0)))
                        evt_idx.append(event)
                        KS_xy.append(cls(torch.stack((x1, y1), axis=-1), torch.stack((x2, y2), axis=-1), alternative))
                        KS_xt.append(cls(torch.stack((x1, t1), axis=-1), torch.stack((x2, t2), axis=-1), alternative))
                        KS_yt.append(cls(torch.stack((y1, t1), axis=-1), torch.stack((y2, t2), axis=-1), alternative))
                        #Kernel_xy.append(kernel_test(x1, y1, x2, y2))
                        #Kernel_xt.append(kernel_test(x1, t1, x2, t2))
                        #Kernel_yt.append(kernel_test(y1, t1, y2, t2))
                    end = time.time()
                    print('time:', end - start)
        df['evt_idx'] = evt_idx
        df['KS_xy'] = KS_xy
        df['KS_xt'] = KS_xt
        df['KS_yt'] = KS_yt
        #df['Kernel_xy'] = Kernel_xy
        #df['Kernel_xt'] = Kernel_xt
        #df['Kernel_yt'] = Kernel_yt
        if use_rand:
            df.to_hdf('./uncertainty/2D/2duncertainty_rand_ndimkolmogorov_{}_{}'.format(photons, i), key = 'df', mode = 'w')
        else:
            df.to_hdf('./uncertainty/2D/2duncertainty_ndimkolmogorov_{}_{}'.format(photons, i), key = 'df', mode = 'w')

nr_of_photons = np.arange(100, 7000, 100)
use_rand = True
cls = ndks2d.ndKS()
alternative = True
for photons in nr_of_photons:
    df = pd.DataFrame(columns = ['evt_idx', 'KS_xy_same', 'KS_xt_same', 'KS_yt_same', 'KS_xy_rand', 'KS_xt_rand', 'KS_yt_rand'])#, 'Kernel_xy', 'Kernel_xt', 'Kernel_yt'])
    KS_xy_same = []
    KS_xt_same = []
    KS_yt_same = []
    KS_xy_rand = []
    KS_xt_rand = []
    KS_yt_rand = []
    evt_idx = []
    for i in range(len(dataframes)):
        for event in event_idx[i]:
            print(event)
            if len(dataframes[i][dataframes[i].evt_idx == event].index) > 10000:
                chosen_event = dataframes[i][dataframes[i].evt_idx == event]
                if (chosen_event.detection_pixel_x.max(axis = 0) - chosen_event.detection_pixel_x.min(axis = 0) != 0) \
                & (chosen_event.detection_pixel_y.max(axis = 0) - chosen_event.detection_pixel_y.min(axis = 0) != 0) \
                & (chosen_event.detection_time.max(axis = 0) - chosen_event.detection_time.min(axis = 0) != 0):
                    draw_rand = np.arange(0, len(chosen_event))
                    rand_event = dataframes[i][dataframes[i].evt_idx == random.choice(event_idx[i])]
                    rand_length = np.arange(0, len(rand_event))
                    start = time.time()
                    for j in range(5):
                        event_sample1 = np.random.choice(draw_rand, photons)
                        x1 = torch.from_numpy(np.asarray(chosen_event.iloc[event_sample1,:].detection_pixel_x.to_numpy() + 23)/46.)
                        y1 = torch.from_numpy((chosen_event.iloc[event_sample1,:].detection_pixel_y.to_numpy() + 4.2)/(4.2+1.01))
                        t1 = torch.from_numpy((chosen_event.iloc[event_sample1,:].detection_time.to_numpy())/(chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0)))
                        event_samplerand = np.random.choice(rand_length, photons)
                        xrand = torch.from_numpy((rand_event.iloc[event_samplerand,:].detection_pixel_x.to_numpy() + 23)/46.)
                        yrand = torch.from_numpy((rand_event.iloc[event_samplerand,:].detection_pixel_y.to_numpy() + 4.2)/(4.2+1.01))
                        trand = torch.from_numpy(rand_event.iloc[event_samplerand,:].detection_time.to_numpy()/(chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0)))
                        event_sample2 = np.random.choice(draw_rand, photons)
                        x2 = torch.from_numpy((chosen_event.iloc[event_sample2,:].detection_pixel_x.to_numpy() + 23)/46.)
                        y2 = torch.from_numpy((chosen_event.iloc[event_sample2,:].detection_pixel_y.to_numpy() + 4.2)/(4.2+1.01))
                        t2 = torch.from_numpy(chosen_event.iloc[event_sample2,:].detection_time.to_numpy()/(chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0)))
                        evt_idx.append(event)
                        KS_xy_same.append(cls(torch.stack((x1, y1), axis=-1), torch.stack((x2, y2), axis=-1), alternative))
                        KS_xt_same.append(cls(torch.stack((x1, t1), axis=-1), torch.stack((x2, t2), axis=-1), alternative))
                        KS_yt_same.append(cls(torch.stack((y1, t1), axis=-1), torch.stack((y2, t2), axis=-1), alternative))
                        KS_xy_rand.append(cls(torch.stack((x1, y1), axis=-1), torch.stack((xrand, yrand), axis=-1), alternative))
                        KS_xt_rand.append(cls(torch.stack((x1, t1), axis=-1), torch.stack((xrand, trand), axis=-1), alternative))
                        KS_yt_rand.append(cls(torch.stack((y1, t1), axis=-1), torch.stack((yrand, trand), axis=-1), alternative))
                        #Kernel_xy.append(kernel_test(x1, y1, x2, y2))
                        #Kernel_xt.append(kernel_test(x1, t1, x2, t2))
                        #Kernel_yt.append(kernel_test(y1, t1, y2, t2))
                    end = time.time()
                    print('time:', end - start)
    df['evt_idx'] = evt_idx
    df['KS_xy_same'] = KS_xy_same
    df['KS_xt_same'] = KS_xt_same
    df['KS_yt_same'] = KS_yt_same
    df['KS_xy_rand'] = KS_xy_rand
    df['KS_xt_rand'] = KS_xt_rand
    df['KS_yt_rand'] = KS_yt_rand
    df.to_hdf('./uncertainty/2D/2duncertainty_all_ndimkolmogorov_{}'.format(photons), key = 'df', mode = 'w')
