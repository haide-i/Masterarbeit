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

df = pd.read_hdf('/ceph/ihaide/photons/first_200/clean_photons_100x1E5_first200_5.h5')
df.head()

photons = np.arange(10, 160, 10)
#photons = np.arange(200, 2900, 100)
#photons2 = np.array((3200, 3500, 3800, 4100))
#photons = np.concatenate((photons1, photons, photons2))
event_nr = 0
for p in photons:
    file = ceph + '3duncertainty_all_ndimkolmogorov_{}'.format(p)
    if os.path.isfile(file):
        df_same = pd.read_hdf(ceph + '3duncertainty_all_first200_ndimkolmogorov_{}'.format(p))
        event_nr = len(df_same.index)
        plt.hist(df_same['KS_xyt_same'], bins = 20, alpha = 0.7, label='same')#, density=True)
        plt.hist(df_same['KS_xyt_rand'], bins = 20, alpha = 0.7, label='random')#, density=True)
        plt.xlabel('Distance')
        plt.ylabel('No of measurements')
        plt.title('3D KS distance with {} photons'.format(p))
        plt.legend()
        #plt.savefig(ceph + '/plots/3d_ndimkolmogorov_dist_allfiles_{}'.format(p))
        plt.show()
print(event_nr)

photons1 = np.arange(10, 200, 10)
photons = np.arange(200, 2900, 100)
photons2 = np.array((3200, 3500, 3800, 4100))
photons = np.concatenate((photons1, photons, photons2))
mean_same = []
mean_rand = []
var_same = []
var_rand = []
x = []
for p in photons:
    file = ceph + '3duncertainty_all_ndimkolmogorov_{}'.format(p)
    if os.path.isfile(file):
        df_same = pd.read_hdf(ceph + '3duncertainty_all_ndimkolmogorov_{}'.format(p))
        x.append(p)
        mean_same.append(df_same['KS_xyt_same'].mean(axis=0))
        mean_rand.append(df_same['KS_xyt_rand'].mean(axis=0))
        var_same.append(df_same['KS_xyt_same'].std(axis=0, ddof=0))
        var_rand.append(df_same['KS_xyt_rand'].std(axis=0, ddof=0))
onesigma_same = np.asarray(var_same)
onesigma_rand = np.asarray(var_rand)

# +
from scipy.interpolate import make_interp_spline, BSpline

x_new = np.linspace(10, 3500, 10000)
spl_same = make_interp_spline(photons, onesigma_same, k=3)
power_smooth = spl_same(photons)
plt.plot(photons, power_smooth, 'r')
plt.plot(photons, onesigma_same, 'b')
plt.show()
# -

plt.figure(figsize=(20, 10))
plt.errorbar(x, mean_same, yerr = onesigma_same, fmt='.b', label = 'Same initial properties')
#plt.fill_between(photons, np.asarray(mean_same) + power_smooth, np.asarray(mean_same) - power_smooth, color = 'b', alpha = 0.5)
#plt.fill_between(photons, np.asarray(mean_same) + 2*onesigma_same, np.asarray(mean_same) - 2*onesigma_same, color = 'b', alpha = 0.3)
plt.errorbar(x, mean_rand, yerr = onesigma_rand, fmt='.r', label = 'Random initial properties')
#plt.fill_between(photons, np.asarray(mean_rand) + onesigma_rand, np.asarray(mean_rand) - onesigma_rand, color = 'r', alpha = 0.5, label = r'1$\sigma$')
#plt.fill_between(photons, np.asarray(mean_rand) + 2*onesigma_rand, np.asarray(mean_rand) - 2*onesigma_rand, color = 'r', alpha = 0.3, label = r'2$\sigma$')
plt.xlabel('No of photons')
plt.ylabel('Mean distance')
plt.title('Distance mean with error')
plt.legend()
plt.savefig(ceph + '/plots/3dKS_1sigmaerror_mean_10_1400')
plt.show()
