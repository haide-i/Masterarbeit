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
home = os.getenv("HOME")
filenames = glob(home + '/data/photons/without_non_det/clean_photons_100x1E5_randpos_randdir_mod5_*.h5')
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


def kolmogorov_1d(y1, y2):
    return np.max(abs(np.cumsum(y1)-np.cumsum(y2)))


def semd(p,t):
    cdf_p = np.cumsum(p)
    cdf_t = np.cumsum(t)
    d_cdf = cdf_p - cdf_t
    d_cdf_s = np.sum(np.power(d_cdf, 2))
    return np.sum(d_cdf_s)


dataframes = [pd.read_hdf(f) for f in filenames]
event_idx = []
for i in range(len(dataframes)):
    event_idx.append(np.unique(dataframes[i].evt_idx))

nr_of_photons = 100
use_rand = True
for j in (100, 7000):
    for i in range(len(dataframes)):
        del df
        df = pd.DataFrame(columns = ['evt_idx', 'EMD_x', 'EMD_y', 'EMD_t', 'EMD_xy', 'EMD_xt', 'EMD_yt'])#, 
                                     #'KS_x', 'KS_y', 'KS_t', 'KS_xy', 'KS_xt', 'KS_yt'])
        EMD_x = []
        EMD_y = []
        EMD_t = []
        EMD_xy = []
        EMD_xt = []
        EMD_yt = []
        #KS_x = []
        #KS_y = []
        #KS_t = []
        #KS_xy = []
        #KS_xt = []
        #KS_yt = []
        evt_idx = []
        for event in event_idx[i]:
            if len(dataframes[i][dataframes[i].evt_idx == event].index) > 10000:
                chosen_event = dataframes[i][dataframes[i].evt_idx == event]
                if use_rand:
                    rand_event = dataframes[i][dataframes[i].evt_idx == random.choice(event_idx[i])]
                    rand_length = np.arange(0, len(rand_event))
                draw_rand = np.arange(0, len(chosen_event))
                for j in range(5):
                    event_sample1 = np.random.choice(draw_rand, j)
                    
                    x1 = (chosen_event.iloc[event_sample1,:].detection_pixel_x + 23)/46.
                    y1 = (chosen_event.iloc[event_sample1,:].detection_pixel_y + 4.2)/(4.2+1.01)
                    t1 = (chosen_event.iloc[event_sample1,:].detection_time)/(chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0))
                    if use_rand:
                        event_sample2 = np.random.choice(rand_length, j)
                        x2 = (rand_event.iloc[event_sample2,:].detection_pixel_x + 23)/46.
                        y2 = (rand_event.iloc[event_sample2,:].detection_pixel_y + 4.2)/(4.2+1.01)
                        t2 = rand_event.iloc[event_sample2,:].detection_time/(chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0))
                    else:
                        event_sample2 = np.random.choice(draw_rand, j)
                        x2 = (chosen_event.iloc[event_sample2,:].detection_pixel_x + 23)/46.
                        y2 = (chosen_event.iloc[event_sample2,:].detection_pixel_y + 4.2)/(4.2+1.01)
                        t2 = chosen_event.iloc[event_sample2,:].detection_time/(chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0))
                    evt_idx.append(event)
                    EMD_x.append(semd(x1, x2))
                    EMD_y.append(semd(y1, y2))
                    EMD_t.append(semd(t1, t2))
                    EMD_xy.append(semd((x1*x1 + y1*y1)**0.5, (x2*x2 + y2*y2)**0.5))
                    EMD_xt.append(semd((x1*x1 + t1*t1)**0.5, (x2*x2 + t2*t2)**0.5))
                    EMD_yt.append(semd((t1*t1 + y1*y1)**0.5, (t2*t2 + y2*y2)**0.5))
                    #KS_x.append(kolmogorov_1d(x1, x2))
                    #KS_y.append(kolmogorov_1d(y1, y2))
                    #KS_t.append(kolmogorov_1d(t1, t2))
                    #KS_xy.append(kolmogorov_1d((x1*x1 + y1*y1)**0.5, (x2*x2 + y2*y2)**0.5))
                    #KS_xt.append(kolmogorov_1d((x1*x1 + t1*t1)**0.5, (x2*x2 + t2*t2)**0.5))
                    #KS_yt.append(kolmogorov_1d((t1*t1 + y1*y1)**0.5, (t2*t2 + y2*y2)**0.5))
        df['evt_idx'] = evt_idx
        df['EMD_x'] = EMD_x
        df['EMD_y'] = EMD_y
        df['EMD_t'] = EMD_t
        df['EMD_xy'] = EMD_xy
        df['EMD_xt'] = EMD_xt
        df['EMD_yt'] = EMD_yt
        #df['KS_x'] = KS_x
        #df['KS_y'] = KS_y
        #df['KS_t'] = KS_t
        #df['KS_xy'] = KS_xy
        #df['KS_xt'] = KS_xt
        #df['KS_yt'] = KS_yt
        if use_rand:
            df.to_hdf('./uncertainty/1D/random_events/1duncertainty_semd_random_{}_{}'.format(j, i), key = 'df', mode = 'w')     
        else:
            df.to_hdf('./uncertainty/1D/same/1duncertainty_semd_{}_{}'.format(j, i), key = 'df', mode = 'w') 

test_df = pd.read_hdf('./uncertainty/1D/same/1duncertainty_7000_1')
test_df.head(-1)
