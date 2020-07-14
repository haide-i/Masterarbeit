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
home = os.getenv("HOME")

filename1 = home + "/data/photons/photons_100x1E5_randpos_randdir_mod5_99.h5"
filename2 = home + "/data/photons/photons_100x1E5_randpos_randdir_mod5_3.h5"

f = h5.File(filename1, 'r')
df = pd.read_hdf(filename1)
#df.head()
print(list(f.keys()))
dset = f['photons']
print(list(dset.keys()))
axis0 = dset['axis0']
axis0.shape
block_0 = dset["block1_items"]
print(block_0.shape)
block_0.dtype
print(block_0[0])

df2 = pd.read_hdf(filename2)
df2.head(-1)

df.head(-1)

event_arr = np.arange(19800, 19899)
for i in event_arr:
    if df[(df.evt_idx == i)].shape[0] != 1:
        event = df[(df.evt_idx == i)]
        prod_x = event.production_x.mean(axis=0)
        prod_y = event.production_y.mean(axis=0)
        prod_z = event.production_z.mean(axis=0)
        print(i, ": x = ", prod_x, " y = ", prod_y, " z = ", prod_z)
        prod_px = event.production_px.mean(axis=0)
        prod_py = event.production_py.mean(axis=0)
        prod_pz = event.production_pz.mean(axis=0)
        print(i, ": px = ", prod_px, " py = ", prod_py, " pz = ", prod_pz)
        max_time = event.detection_time.max(axis = 0)
        time_x = np.linspace(0, max_time, event.shape[0])
        detection_x = event.detection_pixel_x
        detection_y = event.detection_pixel_y
        plt.hist2d(detection_x, detection_y, bins=(100, 100))
        plt.show()
        plt.hist(event.detection_time, bins = 100)
        plt.show()

evt_nbr = []
prod_x = []
prod_y = []
prod_z = []
prod_px = []
prod_py = []
prod_pz = []
for i in range(1, 7):
    filename = home + "/data/photons/photons_100x1E5_randpos_randdir_mod5_{}.h5".format(i)
    df = pd.read_hdf(filename)
    event_arr = np.arange(df["evt_idx"].min(axis=0), df["evt_idx"].max(axis=0))
    for j in event_arr:
        if df[(df.evt_idx == j)].shape[0] != 1:
            event = df[(df.evt_idx == j)]
            if ((abs(event.detection_pixel_x.max(axis=0) - event.detection_pixel_x.min(axis=0)) > 3) 
                or (abs(event.detection_pixel_y.max(axis=0) - event.detection_pixel_y.min(axis=0)) > 3)):  
            #evt_nbr.append(df.evt_idx)
            #event = df[(df.evt_idx == i)]
            #prod_x.append(event.production_x.mean(axis=0))
            #prod_y.append(event.production_y.mean(axis=0))
            #prod_z.append(event.production_z.mean(axis=0))
            #print(i, ": x = ", prod_x, " y = ", prod_y, " z = ", prod_z)
            #prod_px.append(event.production_px.mean(axis=0))
            #prod_py.append(event.production_py.mean(axis=0))
            #prod_pz.append(event.production_pz.mean(axis=0))
            #print(i, ": px = ", prod_px, " py = ", prod_py, " pz = ", prod_pz)
                prod_x = event.production_x.mean(axis=0)
                prod_y = event.production_y.mean(axis=0)
                prod_z = event.production_z.mean(axis=0)
                print(j, ": x = ", prod_x, " y = ", prod_y, " z = ", prod_z)
                prod_px = event.production_px.mean(axis=0)
                prod_py = event.production_py.mean(axis=0)
                prod_pz = event.production_pz.mean(axis=0)
                print(j, ": px = ", prod_px, " py = ", prod_py, " pz = ", prod_pz)
                max_time = event.detection_time.max(axis = 0)
                time_x = np.linspace(0, max_time, event.shape[0])
                detection_x = event.detection_pixel_x
                detection_y = event.detection_pixel_y
                plt.hist2d(detection_x, detection_y, bins=(100, 100))
                plt.show()
                plt.hist(event.detection_time, bins = 100)
                plt.show()

# +

filename = home + "/data/photons/photons_100x1E5_randpos_randdir_mod5_{}.h5".format(2)
df = pd.read_hdf(filename)
event_arr = np.arange(df["evt_idx"].min(axis=0), df["evt_idx"].max(axis=0))
for j in event_arr:
    if j == 448:
        nbr = df[(df.evt_idx == j)].shape[0]
        print(nbr)
        event = df[(df.evt_idx == j)]
        prod_x = event.production_x.mean(axis=0)
        prod_y = event.production_y.mean(axis=0)
        prod_z = event.production_z.mean(axis=0)
        print(j, ": x = ", prod_x, " y = ", prod_y, " z = ", prod_z)
        prod_px = event.production_px.mean(axis=0)
        prod_py = event.production_py.mean(axis=0)
        prod_pz = event.production_pz.mean(axis=0)
        print(j, ": px = ", prod_px, " py = ", prod_py, " pz = ", prod_pz)
        max_time = event.detection_time.max(axis = 0)
        time_x = np.linspace(0, max_time, event.shape[0])
        detection_x = event.detection_pixel_x
        detection_y = event.detection_pixel_y
        plt.figure(figsize=(8, 6))
        plt.hist2d(detection_x, detection_y, bins=(100, 100))
        plt.xlabel("Detection Pixel x")
        plt.ylabel("Detection Pixel y")
        plt.title("x = {}, y = {}, z = {}".format(np.round(prod_x, 2), np.round(prod_y,2), np.round(prod_z,2)))
        plt.savefig("./plots/photons/2Ddetectedxy.pdf")
        plt.show()
        plt.figure(figsize=(8, 6))
        plt.hist(event.detection_time, bins = 100)
        plt.xlabel("Detection time")
        plt.ylabel("Detected Photons")
        plt.savefig("./plots/photons/DetectedTime.pdf")
        plt.show()


# +
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=(20, 2.5))
#ax = fig.gca(projection='3d')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
inside_nbr = 0
total = 0
#for i in range(1, 7):
filename = home + "/data/photons/photons_100x1E5_randpos_randdir_mod5_{}.h5".format(2)
df = pd.read_hdf(filename)
event_arr = np.arange(df["evt_idx"].min(axis=0), df["evt_idx"].max(axis=0))
for j in event_arr:
    event = df[(df.evt_idx == j)]
    prod_x = event.production_x
    prod_y = event.production_y
    prod_z = event.production_z
    det_x = event.detection_pixel_x
    det_y = event.detection_pixel_y
    if (abs(prod_y.mean(axis=0)) <= 10.) & (abs(prod_x.mean(axis=0)) <= 50.):
        inside_nbr += 1
        #ax.scatter(prod_x, prod_y, prod_z)
    plt.plot(det_x, det_y, '.b')
    total += 1
print(total)
print(inside_nbr)
print("ratio = ", float(total)/inside_nbr)
plt.show()
