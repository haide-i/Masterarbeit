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
import gzip
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
home = os.getenv("HOME")
filenames = glob(home + '/data/photons/without_non_det/clean_photons_100x1E5_randpos_randdir_mod5_*.h5')

dataframes = [pd.read_hdf(f) for f in filenames]

event_idx = []
for i in range(len(dataframes)):
    event_idx.append(np.unique(dataframes[i].evt_idx))

print(event_idx)

check = 0
skip = False
for i in range(len(dataframes)):
    if skip:
        break
    for event in event_idx[i]:
        print(event)
        if len(dataframes[i][dataframes[i].evt_idx == event].index) > 50000:
            check += 1
            if check > 2:
                chosen_event = dataframes[i][dataframes[i].evt_idx == event]
                skip = True
                break
print(check)


print(chosen_event.shape)
chosen_event.head(-1)
#plt.plot(chosen_event.detection_pixel_x, chosen_event.detection_pixel_y, '.b')
#plt.plot(chosen_event[chosen_event.detection_time > 3.0].detection_pixel_x, chosen_event[chosen_event.detection_time > 3.0].detection_pixel_y, '.r')
plt.plot(chosen_event[chosen_event.detection_time < 3.0].detection_pixel_x, chosen_event[chosen_event.detection_time < 3.0].detection_pixel_y, '.b')

plt.plot(chosen_event.detection_pixel_x, chosen_event.detection_time, '.b')

event_sample1 = np.random.choice(chosen_event.index, 7000)
event_sample2 = np.random.choice(chosen_event.index, 7000)
event_sample3 = np.random.choice(chosen_event.index, 7000)
event_sample4 = np.random.choice(chosen_event.index, 7000)
event_sample5 = np.random.choice(chosen_event.index, 7000)

plt.plot(chosen_event.iloc[event_sample1,:].detection_pixel_x, chosen_event.iloc[event_sample1,:].detection_pixel_y, '.b')

plt.plot(chosen_event.iloc[event_sample2,:].detection_pixel_x, chosen_event.iloc[event_sample2,:].detection_pixel_y, '.b')

plt.plot(chosen_event.iloc[event_sample3,:].detection_pixel_x, chosen_event.iloc[event_sample3,:].detection_pixel_y, '.b')

plt.plot(chosen_event.iloc[event_sample4,:].detection_pixel_x, chosen_event.iloc[event_sample4,:].detection_pixel_y, '.b')

plt.plot(chosen_event.iloc[event_sample5,:].detection_pixel_x, chosen_event.iloc[event_sample5,:].detection_pixel_y, '.b')
