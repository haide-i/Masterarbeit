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
ogun_dir = '/home/ihaide/Masterarbeit/metrics_test/photon_data/OpticalGun/'
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

datanames = np.arange(0, 1000, 200)
photons = np.arange(10, 160, 10)
for p in photons:
    df = pd.concat([pd.read_hdf(ceph + '3dKS_singletoall_groundevt1_2_70ns_{}_{}_photons{}.h5'.format(first, first+200, p)) \
                         for first in datanames])
    df.to_hdf(ceph+'3dKS_singletoall_groundevt1_2_70ns_photons{}.h5'.format(p), key = 'df', mode='w', complevel=9, complib='blosc:lz4')

photons = np.arange(10, 160, 10)
frames = np.arange(0, 1000, 200)
event_nr = 0
for p in photons:
    file = ogun_dir + '3ddist/ogun_gaussx10_sigma2_photons{}.h5'.format(p)
    if os.path.isfile(file):
        df = pd.read_hdf(file)
        #df_diff = df.drop(df[df.evt_idx == df.rand_evt_idx].index)
        plt.hist(df['KS_3D'], bins = 100, alpha = 0.7)#, density=True)
        #plt.hist(df_diff['KS_xyt_rand'], bins = 20, alpha = 0.7, label='random')#, density=True)
        plt.xlabel('Distance')
        plt.ylabel('No of measurements')
        plt.title('3D KS distance with {} photons'.format(p))
        #plt.legend()
        plt.savefig(ogun_dir + '/plots/ogun_Gaussx10_sigma2_3DKSdist_{}'.format(p))
        plt.show()

# +
photons = np.arange(10, 160, 10)
#photons = np.arange(200, 2900, 100)
#photons2 = np.array((3200, 3500, 3800, 4100))
#photons = np.concatenate((photons1, photons, photons2))
mean_same = []
#mean_rand = []
var_same = []
sigma1 = []
sigma1_var = []
sigma2 = []
sigma2_var = []
sigma0 = []
sigma0_var = []
sigma7 = []
sigma7_var = []
#var_rand = []
max_min = []
equal = []
x = []
for p in photons:
    file = ceph + '3dKS_singletoall_groundevt1_1_70ns_photons{}.h5'.format(p)
    ogun1_file = ogun_dir + '/3ddist/ogun_gaussx10_sigma1_photons{}.h5'.format(p)
    ogun2_file = ogun_dir + '/3ddist/ogun_gaussx10_sigma2_photons{}.h5'.format(p)
    ogun7_file = ogun_dir + '/3ddist/ogun_gaussx10_sigma7_photons{}.h5'.format(p)
    ogun0_file = ogun_dir + '/3ddist/ogun_gaussx10_sigma0_photons{}.h5'.format(p)
    if os.path.isfile(file):
        df_same = pd.read_hdf(file)
        ogun1_df = pd.read_hdf(ogun1_file)
        ogun2_df = pd.read_hdf(ogun2_file)
        ogun7_df = pd.read_hdf(ogun7_file)
        ogun0_df = pd.read_hdf(ogun0_file)
        equal.append((df_same[df_same.rand_evt_idx == 202].KS_3D.mean(axis=0)))
        x.append(p)
        max_min.append((df_same['KS_3D'].astype(float).nsmallest(2)))
        mean_same.append(df_same['KS_3D'].mean(axis=0))
        #mean_rand.append(df_same['KS_xyt_rand'].mean(axis=0))
        var_same.append(df_same['KS_3D'].std(axis=0, ddof=1))
        #var_rand.append(df_same['KS_xyt_rand'].std(axis=0, ddof=1))
        sigma1.append(ogun1_df['KS_3D'].mean(axis=0))
        sigma1_var.append(ogun1_df['KS_3D'].std(axis=0, ddof=1))
        sigma2.append(ogun2_df['KS_3D'].mean(axis=0))
        sigma2_var.append(ogun2_df['KS_3D'].std(axis=0, ddof=1))
        sigma7.append(ogun7_df['KS_3D'].mean(axis=0))
        sigma7_var.append(ogun7_df['KS_3D'].std(axis=0, ddof=1))
        sigma0.append(ogun0_df['KS_3D'].mean(axis=0))
        sigma0_var.append(ogun0_df['KS_3D'].std(axis=0, ddof=1))

onesigma_ogun1 = np.asarray(sigma1_var)
onesigma_ogun2 = np.asarray(sigma2_var)
onesigma_ogun7 = np.asarray(sigma7_var)
onesigma_ogun0 = np.asarray(sigma0_var)
onesigma_same = np.asarray(var_same)
#onesigma_rand = np.asarray(var_rand)
# -

print(equal)
max_minT = np.asarray(max_min).T
max_minT[1]
onesigma_ogun1
#max_min_200 = np.asarray(max_min_200).T

# +
plt.figure(figsize=(20, 10))
plt.ylim((0, 1.1))
#plt.plot(x, mean_same, '.r', label = 'Random')
plt.plot(x, sigma7, '.r', label = r'7$\sigma$ diff')
plt.plot(x, sigma2, '.g', label = r'2$\sigma$ diff')

plt.plot(x, sigma1, '.c', label = r'1$\sigma$ diff')
plt.plot(x, sigma0, '.b', label = r'0.3$\sigma$ diff')
#plt.plot(x, equal, '.k', label = 'Same')
#plt.plot(x, max_minT[1], '-b', alpha = 0.3, label = '1')

#plt.fill_between(photons, np.asarray(mean_same) + onesigma_same, np.asarray(mean_same) - onesigma_same, color = 'r', alpha = 0.3)
plt.fill_between(photons, np.asarray(sigma1) + onesigma_ogun1, np.asarray(sigma1) - onesigma_ogun1, color = 'g', alpha = 0.3)
plt.fill_between(photons, np.asarray(sigma2) + onesigma_ogun2, np.asarray(sigma2) - onesigma_ogun2, color = 'c', alpha = 0.3)
plt.fill_between(photons, np.asarray(sigma0) + onesigma_ogun0, np.asarray(sigma0) - onesigma_ogun0, color = 'b', alpha = 0.3)
plt.fill_between(photons, np.asarray(sigma7) + onesigma_ogun7, np.asarray(sigma7) - onesigma_ogun7, color = 'r', alpha = 0.3)


#plt.plot(x, mean_rand, '.r', label = 'All photons - Random')
#plt.fill_between(photons, np.asarray(mean_rand) + onesigma_rand, np.asarray(mean_rand) - onesigma_rand, color = 'r', alpha = 0.3)#, label = r'2$\sigma$')
plt.xlabel('No of photons')
plt.ylabel('Mean distance')
plt.title('3D KS Distance mean with error')
plt.legend(loc='lower left')
plt.savefig(ogun_dir + '/plots/3dKS_distancemean_withogungaussx10_errorbands.pdf')
plt.show()

