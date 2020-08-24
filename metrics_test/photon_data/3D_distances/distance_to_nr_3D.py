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
cwd = os.getcwd()
ceph = '/ceph/ihaide/distances/3D/'
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
from ekp_style import set_ekp_style
set_ekp_style(set_sizes=True, set_background=True, set_colors=True)

datanames = np.arange(0, 1000, 200)
photons = np.arange(10, 160, 10)
for p in photons:
    df = pd.concat([pd.read_hdf(ceph + '3dKS_singletoall_70ns_{}_{}_photons{}.h5'.format(first, first+200, p)) \
                         for first in datanames])
    df.to_hdf(ceph+'3dKS_singletoall_70ns_photons{}.h5'.format(p), key = 'df', mode='w', complevel=9, complib='blosc:lz4')

df = pd.read_hdf(ceph + '3dKS_singletoall_70ns_photons10.h5')
df.head()

photons = np.arange(10, 160, 10)
frames = np.arange(0, 1000, 200)
event_nr = 0
for p in photons:
    file = ceph + '3dKS_singletoall_70ns_photons{}.h5'.format(p)
    if os.path.isfile(file):
        df = pd.read_hdf(file)
        #df_diff = df.drop(df[df.evt_idx == df.rand_evt_idx].index)
        plt.hist(df['KS_3D'], bins = 100, alpha = 0.7)#, density=True)
        #plt.hist(df_diff['KS_xyt_rand'], bins = 20, alpha = 0.7, label='random')#, density=True)
        plt.xlabel('Distance')
        plt.ylabel('No of measurements')
        plt.title('3D KS distance with {} photons'.format(p))
        plt.legend()
        #plt.savefig(cwd + '/plots/3d_ndimkolmogorov_dist_allfiles_{}'.format(p))
        plt.show()

df_diff.head()

df_new = pd.DataFrame(columns = df_same.columns)
for p in photons:
    file = ceph + '3duncertainty_all_ndimkolmogorov_{}'.format(p)
    if os.path.isfile(file):
        df = pd.read_hdf(file)
        df_diff = df.drop(df[df.evt_idx == df.rand_evt_idx].index)
        df_new = df_new.append(df_diff[df_diff['KS_xyt_rand'] < 0.5])

print(df_new.shape)
print(df_new['KS_xyt_rand'].min(axis=0))
df_new

file88 = pd.read_hdf('/ceph/ihaide/photons/without_non_detected/clean_photons_100x1E5_randpos_randdir_mod5_88.h5')
event88 = file88[file88.evt_idx == 17640]
file555 = pd.read_hdf('/ceph/ihaide/photons/without_non_detected/clean_photons_100x1E5_randpos_randdir_mod5_555.h5')
event555 = file555[file555.evt_idx == 111074]

event88.head()

prod_x = event88.production_x.mean(axis=0)
prod_y = event88.production_y.mean(axis=0)
prod_z = event88.production_z.mean(axis=0)
print(" x = ", prod_x, " y = ", prod_y, " z = ", prod_z)
prod_px = event88.production_px.mean(axis=0)
prod_py = event88.production_py.mean(axis=0)
prod_pz = event88.production_pz.mean(axis=0)
print(": px = ", prod_px, " py = ", prod_py, " pz = ", prod_pz)
max_time = event88.detection_time.max(axis = 0)
time_x = np.linspace(0, max_time, event88.shape[0])
detection88_x = event88.detection_pixel_x
detection88_y = event88.detection_pixel_y
plt.hist2d(detection_x, detection_y, bins=(100, 100))
plt.title('X-Y')
plt.savefig(cwd + 'event88_xy.pdf')
plt.show()
plt.hist(event88.detection_time, bins = 100)
plt.title('Time')
plt.savefig(cwd + 'event88_t.pdf')
plt.show()
plt.figure(figsize = (16, 2))
x_start = event88.production_x.mean(axis=0)
y_start = event88.production_y.mean(axis=0)
z_start = event88.production_z.mean(axis=0)
x_mom = event88.production_px.mean(axis=0)
y_mom = event88.production_py.mean(axis=0)
z_mom = event88.production_pz.mean(axis=0)
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

prod_x = event555.production_x.mean(axis=0)
prod_y = event555.production_y.mean(axis=0)
prod_z = event555.production_z.mean(axis=0)
print(" x = ", prod_x, " y = ", prod_y, " z = ", prod_z)
prod_px = event555.production_px.mean(axis=0)
prod_py = event555.production_py.mean(axis=0)
prod_pz = event555.production_pz.mean(axis=0)
print(": px = ", prod_px, " py = ", prod_py, " pz = ", prod_pz)
max_time = event555.detection_time.max(axis = 0)
time_x = np.linspace(0, max_time, event555.shape[0])
detection_x = event555.detection_pixel_x
detection_y = event555.detection_pixel_y
plt.hist2d(detection_x, detection_y, bins=(100, 100))
plt.title('X-Y')
plt.savefig(cwd + 'event555_xy.pdf')
plt.show()
plt.hist(event555.detection_time, bins = 100)
plt.title('Time')
plt.savefig(cwd + 'event555_t.pdf')
plt.show()
plt.figure(figsize = (16, 2))
x_start = event555.production_x.mean(axis=0)
y_start = event555.production_y.mean(axis=0)
z_start = event555.production_z.mean(axis=0)
x_mom = event555.production_px.mean(axis=0)
y_mom = event555.production_py.mean(axis=0)
z_mom = event555.production_pz.mean(axis=0)
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

event555.head()

photons = np.arange(10, 160, 10)
#photons = np.arange(200, 2900, 100)
#photons2 = np.array((3200, 3500, 3800, 4100))
#photons = np.concatenate((photons1, photons, photons2))
mean_same = []
#mean_rand = []
var_same = []
#var_rand = []
max_min = []
x = []
for p in photons:
    file = ceph + '3dKS_singletoall_70ns_photons{}.h5'.format(p)
    #file200 = ceph + '3duncertainty_all_first200_ndimkolmogorov_{}'.format(p)
    if os.path.isfile(file):
        df_same = pd.read_hdf(file)
        #df_same = df.drop(df[df.evt_idx == df.rand_evt_idx].index)
        #df_200 = pd.read_hdf(file200)
        x.append(p)
        max_min.append((df_same['KS_3D'].astype(float).nlargest(3), df_same['KS_3D'].astype(float).nsmallest(3)))
        #max_min_200.append((df_200['KS_xyt_same'].max(axis=0), df_200['KS_xyt_same'].min(axis=0), df_200['KS_xyt_rand'].max(axis=0), df_200['KS_xyt_rand'].min(axis=0)))
        mean_same.append(df_same['KS_3D'].mean(axis=0))
        #mean_rand.append(df_same['KS_xyt_rand'].mean(axis=0))
        var_same.append(df_same['KS_3D'].std(axis=0, ddof=1))
        #var_rand.append(df_same['KS_xyt_rand'].std(axis=0, ddof=1))
        #mean_same200.append(df_200['KS_xyt_same'].mean(axis=0))
        #mean_rand200.append(df_200['KS_xyt_rand'].mean(axis=0))
        #var_same200.append(df_200['KS_xyt_same'].std(axis=0, ddof=1))
        #var_rand200.append(df_200['KS_xyt_rand'].std(axis=0, ddof=1))
#onesigma_same200 = np.asarray(var_same200)
#onesigma_rand200 = np.asarray(var_rand200)
onesigma_same = np.asarray(var_same)
#onesigma_rand = np.asarray(var_rand)

# +

#max_min = np.asarray(max_min).T
max_min[2][1]
#max_min_200 = np.asarray(max_min_200).T
# -

plt.figure(figsize=(20, 10))
plt.plot(x, mean_same, '.b', label = 'All photons - Same')
#plt.plot(x, max_min[0][0], '-b', alpha = 0.3, label = '1')
#plt.plot(x, max_min[0][1], '-r', alpha = 0.3, label = '1')
#plt.plot(x, max_min[1][1], '-r', alpha = 0.2, label = '2')
#plt.plot(x, max_min[2][1], '-r', alpha = 0.1, label = '3')
plt.fill_between(photons, np.asarray(mean_same) + onesigma_same, np.asarray(mean_same) - onesigma_same, color = 'b', alpha = 0.3)
#plt.plot(x, mean_rand, '.r', label = 'All photons - Random')
#plt.fill_between(photons, np.asarray(mean_rand) + onesigma_rand, np.asarray(mean_rand) - onesigma_rand, color = 'r', alpha = 0.3)#, label = r'2$\sigma$')
plt.xlabel('No of photons')
plt.ylabel('Mean distance')
plt.title('3D KS Distance mean with error')
plt.legend(loc='right')
#plt.savefig(cwd + '/plots/3dKS_1sigmaerror_mean_errorbands.pdf')
plt.show()
#plt.figure(figsize=(20, 10))
#plt.plot(x, mean_same200, '.b', label = '200 photons - Same')
#plt.plot(x, max_min_200[0], '-b', alpha = 0.3)
#plt.plot(x, max_min_200[3], '-r', alpha = 0.3)
#plt.fill_between(photons, np.asarray(mean_same200) + onesigma_same200, np.asarray(mean_same200) - onesigma_same200, color = 'b', alpha = 0.3)
#plt.plot(x, mean_rand200, '.r', label = '200 photons - Random')
#plt.fill_between(photons, np.asarray(mean_rand200) + onesigma_rand200, np.asarray(mean_rand200) - onesigma_rand200, color = 'r', alpha = 0.3)#, label = r'2$\sigma$')
#plt.xlabel('No of photons')
#plt.ylabel('Mean distance')
#plt.title('3D KS Distance mean with error')
#plt.legend(loc='right')
#plt.show()
