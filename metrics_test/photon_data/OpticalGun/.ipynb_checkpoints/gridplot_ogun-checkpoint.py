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
import pandas as pd
import os
import sys
sys.path.append('/home/ihaide/Documents/Masterarbeit-master/metrics_test/photon_data')
from class_distance import get_dstc
ceph = '/ceph/ihaide/ogun/Gauss/'

var = 'theta'
sigma_name = np.arange(0, 100, 2)
mu_diff = np.arange(0, 50, 5)
dstc = get_dstc()
photon_nr = (10, 50, 150)
keys = ['Sigma', 'Mu', 'KS_distance', 'momentum_distance', 'position_distance']
distance_df = pd.DataFrame(columns=keys)
distance = []
mu_sigma = []
no_exist = []
not_detected = []
ground_file = ceph + f'ogun_gauss_{var}_mu0_sigma0_1.h5'
ground_evt = pd.read_hdf(ground_file)
for p in photon_nr:
    for sigma in sigma_name:
        for mu in mu_diff:
            if sigma == 0 and mu == 0:
                groundfile_nr = 1
                add = 18
            else:
                groundfile_nr = sigma*100 + mu*4
                add = 19
            print('sigma: ', sigma, ' mu: ', mu, ' file: ', groundfile_nr)
            end_file = ceph + f'ogun_gauss_{var}_mu{mu}_sigma{sigma}_{groundfile_nr + add}.h5'
            if os.path.isfile(end_file):
                if ground_evt.detection_time.mean(axis=0) > 0:
                    print(groundfile_nr)
                    for i in range(1, 20):
                        file_nr = sigma*100 + mu*4 + i
                        new_file = ceph + f'ogun_gauss_{var}_mu{mu}_sigma{sigma}_{file_nr}.h5'
                        if os.path.isfile(new_file):
                            evt2 = pd.read_hdf(ceph + f'ogun_gauss_{var}_mu{mu}_sigma{sigma}_{file_nr}.h5')
                            if evt2.detection_time.mean(axis=0) > 0:
                                distance.append(dstc(ground_evt, evt2, photon_nr))
                                mu_sigma.append((sigma, mu))
    distance_new = np.asarray(distance).T
    mu_sigma_new = np.asarray(mu_sigma).T
    all_data = np.vstack((mu_sigma_new, distance_new))
    for idx, key in enumerate(keys):
        distance_df[key] = all_data[idx]
    distance_df.to_hdf(ceph + f'distance/ogun_groundfl_{var}_dist_photons{photon_nr}.h5', key = 'distance_df', mode = 'w', complevel=9, complib='blosc:lz4')

df = pd.read_hdf(ceph + 'ogun_gauss_x_mu40_sigma58_5975.h5')
df
