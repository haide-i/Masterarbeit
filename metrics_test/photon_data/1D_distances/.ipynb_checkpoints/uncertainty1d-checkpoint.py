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
import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import random
import time
from scipy.stats import wasserstein_distance
home = os.getenv("HOME")
filenames = glob('/ceph/ihaide/photons/without_non_det/clean_photons_100x1E5_randpos_randdir_mod5_*.h5')
cwd = os.getcwd()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


def kolmogorov_1d(y1, y2):
    return np.max(abs(np.cumsum(y1)-np.cumsum(y2)))


def semd(p,t):
    p = np.asarray(p)
    t = np.asarray(t)
    cdf_p = np.cumsum(p)/p.size
    cdf_t = np.cumsum(t)/t.size
    d_cdf = cdf_p - cdf_t
    d_cdf_s = np.sum(np.power(d_cdf, 2))
    return np.mean(d_cdf_s)


def semd_torch(p, )


def emd(u_values, v_values):
    u_values = np.asarray(u_values, dtype=float)
    v_values = np.asarray(v_values, dtype=float)
    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    all_values = np.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')

    deltas = np.diff(all_values)

    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    u_cdf = u_cdf_indices / u_values.size
    
    v_cdf = v_cdf_indices / v_values.size

    return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))


dataframes = [pd.read_hdf(f) for f in filenames]
event_idx = []
for i in range(len(dataframes)):
    event_idx.append(np.unique(dataframes[i].evt_idx))


nr_of_photons = 100
use_rand = True
photons1 = np.arange(10, 200, 10)
photons = np.arange(200, 5000, 200)
photons = np.concatenate((photons1, photons))
for j in photons:
    for i in range(len(dataframes)):
        df_emd = pd.DataFrame(columns = ['evt_idx', 'EMD_x', 'EMD_y', 'EMD_t', 'EMD_xy', 'EMD_xt', 'EMD_yt'])
        dfrand_emd = pd.DataFrame(columns = ['evt_idx', 'EMD_x', 'EMD_y', 'EMD_t', 'EMD_xy', 'EMD_xt', 'EMD_yt'])
        df_semd = pd.DataFrame(columns = ['evt_idx', 'SEMD_x', 'SEMD_y', 'SEMD_t', 'SEMD_xy', 'SEMD_xt', 'SEMD_yt'])
        dfrand_semd = pd.DataFrame(columns = ['evt_idx', 'SEMD_x', 'SEMD_y', 'SEMD_t', 'SEMD_xy', 'SEMD_xt', 'SEMD_yt'])
        EMD_x = []
        EMD_y = []
        EMD_t = []
        EMD_xy = []
        EMD_xt = []
        EMD_yt = []
        SEMD_x = []
        SEMD_y = []
        SEMD_t = []
        SEMD_xy = []
        SEMD_xt = []
        SEMD_yt = []
        randEMD_x = []
        randEMD_y = []
        randEMD_t = []
        randEMD_xy = []
        randEMD_xt = []
        randEMD_yt = []
        randSEMD_x = []
        randSEMD_y = []
        randSEMD_t = []
        randSEMD_xy = []
        randSEMD_xt = []
        randSEMD_yt = []
        evt_idx = []
        for event in event_idx[i]:
            if len(dataframes[i][dataframes[i].evt_idx == event].index) > 10000:
                chosen_event = dataframes[i][dataframes[i].evt_idx == event]
                rand_event = dataframes[i][dataframes[i].evt_idx == random.choice(event_idx[i])]
                rand_length = np.arange(0, len(rand_event))
                draw_rand = np.arange(0, len(chosen_event))
                #print(event)
                start = time.time()
                for k in range(5):
                    event_sample1 = np.random.choice(draw_rand, j)
                    x1 = (chosen_event.iloc[event_sample1,:].detection_pixel_x + 23)/46.
                    #x1 = x1/(np.sum(x1))
                    y1 = (chosen_event.iloc[event_sample1,:].detection_pixel_y + 4.2)/(4.2+1.01)
                    #y1 = y1/(np.sum(y1))
                    t1 = (chosen_event.iloc[event_sample1,:].detection_time)/(chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0))
                    event_samplerand = np.random.choice(rand_length, j)
                    #t1 = t1/(np.sum(t1))
                    xrand = (rand_event.iloc[event_samplerand,:].detection_pixel_x + 23)/46.
                    #xrand = xrand/(np.sum(xrand))
                    yrand = (rand_event.iloc[event_samplerand,:].detection_pixel_y + 4.2)/(4.2+1.01)
                    #yrand = yrand/(np.sum(yrand))
                    trand = rand_event.iloc[event_samplerand,:].detection_time/(chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0))
                    #trand = trand/(np.sum(trand))
                    event_sample2 = np.random.choice(draw_rand, j)
                    x2 = (chosen_event.iloc[event_sample2,:].detection_pixel_x + 23)/46.
                    #x2 = x2/(np.sum(x2))
                    y2 = (chosen_event.iloc[event_sample2,:].detection_pixel_y + 4.2)/(4.2+1.01)
                    #y2 = y2/(np.sum(y2))
                    t2 = chosen_event.iloc[event_sample2,:].detection_time/(chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0))
                    #t2 = t2/(np.sum(t2))
                    evt_idx.append(event)
                    EMD_x.append(emd(x1, x2))
                    EMD_y.append(emd(y1, y2))
                    EMD_t.append(emd(t1, t2))
                    EMD_xy.append(emd((x1*x1 + y1*y1)**0.5, (x2*x2 + y2*y2)**0.5))
                    EMD_xt.append(emd((x1*x1 + t1*t1)**0.5, (x2*x2 + t2*t2)**0.5))
                    EMD_yt.append(emd((t1*t1 + y1*y1)**0.5, (t2*t2 + y2*y2)**0.5))
                    SEMD_x.append(semd(x1, x2))
                    SEMD_y.append(semd(y1, y2))
                    SEMD_t.append(semd(t1, t2))
                    SEMD_xy.append(semd((x1*x1 + y1*y1)**0.5, (x2*x2 + y2*y2)**0.5))
                    SEMD_xt.append(semd((x1*x1 + t1*t1)**0.5, (x2*x2 + t2*t2)**0.5))
                    SEMD_yt.append(semd((t1*t1 + y1*y1)**0.5, (t2*t2 + y2*y2)**0.5))
                    randEMD_x.append(emd(x1, xrand))
                    randEMD_y.append(emd(y1, yrand))
                    randEMD_t.append(emd(t1, trand))
                    randEMD_xy.append(emd((x1*x1 + y1*y1)**0.5, (xrand*xrand + yrand*yrand)**0.5))
                    randEMD_xt.append(emd((x1*x1 + t1*t1)**0.5, (xrand*xrand + trand*trand)**0.5))
                    randEMD_yt.append(emd((t1*t1 + y1*y1)**0.5, (trand*trand + trand*trand)**0.5))
                    randSEMD_x.append(semd(x1, xrand))
                    randSEMD_y.append(semd(y1, yrand))
                    randSEMD_t.append(semd(t1, trand))
                    randSEMD_xy.append(semd((x1*x1 + y1*y1)**0.5, (xrand*xrand + yrand*yrand)**0.5))
                    randSEMD_xt.append(semd((x1*x1 + t1*t1)**0.5, (xrand*xrand + trand*trand)**0.5))
                    randSEMD_yt.append(semd((t1*t1 + y1*y1)**0.5, (trand*trand + yrand*yrand)**0.5))
                end = time.time()
                print('time: ', end - start)
    df_emd['evt_idx'] = evt_idx
    dfrand_emd['evt_idx'] = evt_idx
    df_semd['evt_idx'] = evt_idx
    dfrand_semd['evt_idx'] = evt_idx
    df_emd['EMD_x'] = EMD_x
    df_emd['EMD_y'] = EMD_y
    df_emd['EMD_t'] = EMD_t
    df_emd['EMD_xy'] = EMD_xy
    df_emd['EMD_xt'] = EMD_xt
    df_emd['EMD_yt'] = EMD_yt
    df_semd['SEMD_x'] = SEMD_x
    df_semd['SEMD_y'] = SEMD_y
    df_semd['SEMD_t'] = SEMD_t
    df_semd['SEMD_xy'] = SEMD_xy
    df_semd['SEMD_xt'] = SEMD_xt
    df_semd['SEMD_yt'] = SEMD_yt
    dfrand_emd['EMD_x'] = randEMD_x
    dfrand_emd['EMD_y'] = randEMD_y
    dfrand_emd['EMD_t'] = randEMD_t
    dfrand_emd['EMD_xy'] = randEMD_xy
    dfrand_emd['EMD_xt'] = randEMD_xt
    dfrand_emd['EMD_yt'] = randEMD_yt
    dfrand_semd['SEMD_x'] = randSEMD_x
    dfrand_semd['SEMD_y'] = randSEMD_y
    dfrand_semd['SEMD_t'] = randSEMD_t
    dfrand_semd['SEMD_xy'] = randSEMD_xy
    dfrand_semd['SEMD_xt'] = randSEMD_xt
    dfrand_semd['SEMD_yt'] = randSEMD_yt
    dfrand_emd.to_hdf('./uncertainty/1D/random_events/1duncertainty_size_emd_random_{}'.format(j), key = 'dfrand_emd', mode = 'w')     
    df_emd.to_hdf('./uncertainty/1D/same/1duncertainty_size_emd_{}'.format(j), key = 'df_emd', mode = 'w')
    dfrand_semd.to_hdf('./uncertainty/1D/random_events/1duncertainty_size_semd_random_{}'.format(j), key = 'dfrand_semd', mode = 'w')     
    df_semd.to_hdf('./uncertainty/1D/same/1duncertainty_size_semd_{}'.format(j), key = 'df_semd', mode = 'w') 
