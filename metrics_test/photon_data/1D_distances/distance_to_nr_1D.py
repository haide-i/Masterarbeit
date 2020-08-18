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
ceph = '/ceph/ihaide/distances/1D/'
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
from ekp_style import set_ekp_style
set_ekp_style(set_sizes=True, set_background=True, set_colors=True)

keys = ('x', 'y', 't', 'xy', 'xt', 'yt')
photons = np.concatenate((np.arange(20, 200, 10), np.arange(200, 1500, 200)))
for p in photons:
    emd_same = pd.read_hdf(ceph + '1duncertainty_emd_{}.h5'.format(p))
    emd_random = pd.read_hdf(ceph + '1duncertainty_emd_random_{}.h5'.format(p))
    semd_same = pd.read_hdf(ceph + '1duncertainty_semd_{}.h5'.format(p))
    semd_random = pd.read_hdf(ceph + '1duncertainty_semd_random_{}.h5'.format(p))
    for dist in keys:
        plt.figure(figsize = (20,5))
        plt.subplot(121)
        plt.hist(emd_same[dist], bins = 20, alpha = 0.7, label='same')#, density=True)
        plt.hist(emd_random[dist], bins = 20, alpha = 0.7, label='random')#, density=True)
        plt.title('{} - {}'.format(p, dist))
        plt.legend()
        plt.subplot(122)
        plt.hist(semd_same[dist], bins = 20, alpha = 0.7, label='same SEMD')
        plt.hist(semd_random[dist], bins = 20, alpha = 0.7, label='random SEMD')
        plt.title('{} - {}'.format(p, dist))
        plt.legend()
        #plt.savefig(cwd + '/uncertainty/1D/plots/1duncertainty_norm_emd_semd_{}_{}'.format(p, dist))
        plt.show()

keys = ('x', 'y', 't', 'xy', 'xt', 'yt')
photons = np.concatenate((np.arange(20, 200, 10), np.arange(200, 1500, 200)))
for p in photons:
    emd_same = pd.read_hdf(ceph + '1duncertainty_size_emd_{}.h5'.format(p))
    emd_random = pd.read_hdf(ceph + '1duncertainty_size_emd_random_{}.h5'.format(p))
    semd_same = pd.read_hdf(ceph + '1duncertainty_size_semd_{}.h5'.format(p))
    semd_random = pd.read_hdf(ceph + '1duncertainty_size_semd_random_{}.h5'.format(p))
    for dist in keys:
        plt.figure(figsize = (20,5))
        plt.subplot(121)
        plt.hist(emd_same[dist], bins = 20, alpha = 0.7, label='same')#, density=True)
        plt.title('{} - {}'.format(p, dist))
        plt.legend()
        plt.subplot(122)
        plt.hist(emd_random[dist], bins = 20, alpha = 0.7, label='same SEMD')
        plt.title('{} - {}'.format(p, dist))
        plt.legend()
        #plt.savefig(cwd + '/uncertainty/1D/plots/1duncertainty_norm_onlysame_emd_semd_{}_{}'.format(p, dist))
        plt.show()
        #print(df_same[new_key].head())
        #print(df_random[new_key].head())

# +
mean_emd = dict.fromkeys(keys)
mean_semd = dict.fromkeys(keys)
mean_emd_random = dict.fromkeys(keys)
mean_semd_random = dict.fromkeys(keys)

for dist in keys:
    help_mean_emd = []
    help_mean_semd = []
    help_mean_emd_random = []
    help_mean_semd_random = []
    for p in photons:
        emd_same = pd.read_hdf(ceph + '1duncertainty_size_emd_{}.h5'.format(p))
        emd_random = pd.read_hdf(ceph + '1duncertainty_size_emd_random_{}.h5'.format(p))
        semd_same = pd.read_hdf(ceph + '1duncertainty_size_semd_{}.h5'.format(p))
        semd_random = pd.read_hdf(ceph + '1duncertainty_size_semd_random_{}.h5'.format(p))
        help_mean_emd.append(emd_same[dist].std(axis=0))
        help_mean_semd.append(semd_same[dist].std(axis=0))
        help_mean_emd_random.append(emd_random[dist].std(axis=0))
        help_mean_semd_random.append(semd_random[dist].std(axis=0))
    mean_emd[dist] = help_mean_emd
    mean_semd[dist] = help_mean_semd
    mean_emd_random[dist] = help_mean_emd_random
    mean_semd_random[dist] = help_mean_semd_random
# -

mean_emd

x = np.arange(200, 5000, 200)
for dist in keys:
    plt.figure(figsize = (20,5))
    plt.subplot(121)
    plt.plot(mean_emd[dist], '.b', label='same')
    plt.plot(mean_emd_random[dist], '.r', label='random')
    plt.title('EMD {}'.format(dist))
    plt.legend()
    plt.subplot(122)
    plt.plot(mean_semd[dist], '.b', label='same')
    plt.plot(mean_semd_random[dist], '.r', label='random')
    plt.title('SEMD {}'.format(dist))
    plt.legend()
    plt.show()

for dist in all_keys:
    plt.plot(photons, mean_df[dist], '.b', label = 'same')
    plt.title('{}'.format(dist))
    plt.legend()
    plt.savefig(cwd + '/uncertainty/1D/plots/1duncertainty_norm_emd_semd_mean_onlysame_{}'.format(dist))
    plt.show()
