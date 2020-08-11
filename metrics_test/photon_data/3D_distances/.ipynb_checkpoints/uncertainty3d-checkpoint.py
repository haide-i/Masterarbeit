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

import torch
import numpy as np
import os
import pandas as pd
#import matplotlib.pyplot as plt
from glob import glob
import random
import time
import ndks3d
#from scipy.stats import wasserstein_distance
home = os.getenv("HOME")
filenames = '/ceph/ihaide/photons/without_non_detected/'
cwd = os.getcwd()
#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))

torch.set_num_threads(12)
datanames = np.arange(0, 1000, 1)
evt_dict = dict.fromkeys(datanames)
event_idx = []
for i in datanames:
    print(i)
    event_idx_help = []
    file = filenames + 'clean_photons_100x1E5_randpos_randdir_mod5_{}.h5'.format(i)
    if os.path.isfile(file):
        file = pd.read_hdf(filenames + 'clean_photons_100x1E5_randpos_randdir_mod5_{}.h5'.format(i))
        for event in np.unique(file.evt_idx):
            if len(file[file.evt_idx == event].index) > 10000:
                event_idx_help.append(event)
                event_idx.append((event, i))
    evt_dict[i] = event_idx_help

use_rand = False
cls = ndks3d.ndKS()
alternative = False 
photon_nr = np.arange(10, 200, 10)
photon2 = np.arange(200, 3000, 100)
photon_nr = np.concatenate((photon_nr, photon2))
for photons in photon_nr:
    df = pd.DataFrame(columns = ['evt_idx', 'KS_xyt_same', 'KS_xyt_rand'])
    KS_xyt_same = []
    KS_xyt_rand = []
    evt_idx = []
    for i in range(len(dataframes)):
        for event in event_idx[i]:
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
                    KS_xyt_same.append(cls(torch.stack((x1, y1, t1), axis=-1), torch.stack((x2, y2, t2), axis=-1), alternative).numpy())
                    KS_xyt_rand.append(cls(torch.stack((x1, y1, t1), axis=-1), torch.stack((xrand, yrand, trand), axis=-1), alternative).numpy())
                end = time.time()
                print('time:', end - start)
    df['evt_idx'] = evt_idx
    df['KS_xyt_same'] = KS_xyt_same
    df['KS_xyt_rand'] = KS_xyt_rand
    df.to_hdf('/ceph/ihaide/distances/3D/3duncertainty_all_ndimkolmogorov_{}'.format(photons), key = 'df', mode = 'w')
