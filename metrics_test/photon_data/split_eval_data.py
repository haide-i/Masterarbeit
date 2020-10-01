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
import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
home = os.getenv("HOME")
filenames = glob(home + '/data/photons/without_non_det/clean_photons_100x1E5_randpos_randdir_mod5_*.h5')
import matplotlib.colors as mcolors
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

dataframes = [pd.read_hdf(f) for f in filenames]

event_idx = []
for i in range(len(dataframes)):
    event_idx.append(np.unique(dataframes[i].evt_idx))

check = 0
skip = False
for i in range(len(dataframes)):
    if skip:
        break
    for event in event_idx[i]:
        if event == 406:
            chosen_event = dataframes[i][dataframes[i].evt_idx == event]
        if len(dataframes[i][dataframes[i].evt_idx == event].index) > 50000:
            check += 1
            if check > 17:
                random_event = dataframes[i][dataframes[i].evt_idx == event]
                skip = True
                break
plt.xlim(-25, 25)
plt.plot(chosen_event.detection_pixel_x, chosen_event.detection_time, '.b')
plt.show()
axis = [-25, 25, -4, 1]
plt.figure(figsize=(20, 2.5))
plt.axis(axis)
plt.hist2d(chosen_event.detection_pixel_x, chosen_event.detection_pixel_y, bins=(50,50))#, norm=mcolors.LogNorm(vmin=1), cmap='PuBu_r')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./plots/photons/event406_xyhist.pdf')
plt.show()
plt.hist(chosen_event.detection_time, bins = 100)
plt.yscale('log')
plt.xlabel('Detection time in ns')
plt.ylabel('Counts')
plt.savefig('./plots/photons/event406_detection_time.pdf')
plt.show()


plt.figure(figsize = (16, 2))
x_start = chosen_event.production_x.mean(axis=0)
y_start = chosen_event.production_y.mean(axis=0)
z_start = chosen_event.production_z.mean(axis=0)
x_mom = chosen_event.production_px.mean(axis=0)
y_mom = chosen_event.production_py.mean(axis=0)
z_mom = chosen_event.production_pz.mean(axis=0)
print(x_start, y_start, z_start)
print(x_mom, y_mom, z_mom)
total_p = (x_mom**2 + y_mom**2 + z_mom**2)**0.5
print(total_p)
plt.axis([-25, 25, -4, 1])
plt.plot(x_start, y_start, '.b')
plt.arrow(x_start, y_start, x_mom, y_mom)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.figure(figsize = (16, 2))
plt.axis([-140, 140, -25, 25])
plt.plot(z_start, x_start, '.b')
plt.arrow(z_start, x_start, z_mom, x_mom)
plt.xlabel('z')
plt.ylabel('x')
plt.show()
plt.figure(figsize = (16, 2))
plt.axis([-140, 140, -4, 1])
plt.plot(z_start, y_start, '.b')
plt.arrow(z_start, y_start, z_mom, y_mom)
plt.xlabel('z')
plt.ylabel('y')
plt.show()

axis = [-25, 25, -41, 1]
#plt.plot(chosen_event.detection_pixel_x, chosen_event.detection_pixel_y, '.b')
plt.plot(chosen_event[chosen_event.detection_time > 3.0].detection_pixel_x, chosen_event[chosen_event.detection_time > 3.0].detection_pixel_y, '.r')
#plt.show()
plt.plot(chosen_event[chosen_event.detection_time < 3.0].detection_pixel_x, chosen_event[chosen_event.detection_time < 3.0].detection_pixel_y, '.b')

plt.xlim(-25, 25)
plt.plot(chosen_event.detection_pixel_x, chosen_event.detection_time, '.b')

draw_rand = np.arange(0, len(chosen_event))
draw_rand2 = np.arange(0, len(random_event))
nr_of_photons = 100
event_sample1 = np.random.choice(draw_rand, nr_of_photons)
event_sample2 = np.random.choice(draw_rand, nr_of_photons)
event_sample3 = np.random.choice(draw_rand2, nr_of_photons)
event_sample4 = np.random.choice(draw_rand, nr_of_photons)
event_sample5 = np.random.choice(draw_rand, nr_of_photons)

axis = [-25, 25, -4, 1]
plt.plot(chosen_event.iloc[event_sample1,:].detection_pixel_x, chosen_event.iloc[event_sample1,:].detection_pixel_y, '.b')
plt.axis(axis)

plt.figure(figsize=(20, 2.5))
plt.hist2d(chosen_event.iloc[event_sample2,:].detection_pixel_x, chosen_event.iloc[event_sample2,:].detection_pixel_y, bins = 100)#, norm = mcolors.LogNorm())
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./plots/photons/event406_xyhist_100_1.png')
plt.show()

# +
plt.figure(figsize=(20, 2.5))
plt.hist2d(chosen_event.iloc[event_sample1,:].detection_pixel_x, chosen_event.iloc[event_sample1,:].detection_pixel_y, bins = 100)#, norm = mcolors.LogNorm())
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./plots/photons/event406_xyhist_100_2.png')

plt.show()

# +
plt.figure(figsize=(20, 2.5))
plt.hist2d(random_event.iloc[event_sample3,:].detection_pixel_x, random_event.iloc[event_sample3,:].detection_pixel_y, bins = 100)#, norm = mcolors.LogNorm())
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./plots/photons/rand_event_xyhist_100_1.png')

plt.show()
# -

plt.plot(chosen_event.iloc[event_sample4,:].detection_pixel_x, chosen_event.iloc[event_sample4,:].detection_pixel_y, '.b')
plt.axis(axis)

plt.plot(chosen_event.iloc[event_sample5,:].detection_pixel_x, chosen_event.iloc[event_sample5,:].detection_pixel_y, '.b')
plt.axis(axis)

plt.scatter(chosen_event.detection_pixel_x, chosen_event.detection_pixel_y, c = chosen_event.detection_time, cmap = 'autumn')
plt.colorbar()


