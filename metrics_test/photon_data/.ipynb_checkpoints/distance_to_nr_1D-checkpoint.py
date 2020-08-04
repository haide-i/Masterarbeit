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
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
from ekp_style import set_ekp_style
set_ekp_style(set_sizes=True, set_background=True, set_colors=True)

emd_keys = ('EMD_x', 'EMD_y', 'EMD_t', 'EMD_xy', 'EMD_xt', 'EMD_yt')
semd_keys = ('SEMD_x', 'SEMD_y', 'SEMD_t', 'SEMD_xy', 'SEMD_xt', 'SEMD_yt')
photons = np.arange(100, 2600, 200)
for p in photons:
    df_same = pd.read_hdf(cwd + '/uncertainty/1D/same/1duncertainty_norm_emd_semd_{}'.format(p))
    df_random = pd.read_hdf(cwd + '/uncertainty/1D/random_events/1duncertainty_norm_emd_semd_random_{}'.format(p))
    print(p)
    for dist in emd_keys:
        plt.figure(figsize = (20,5))
        plt.subplot(121)
        plt.hist(df_same[dist], bins = 20, alpha = 0.7, label='same')#, density=True)
        plt.hist(df_random[dist], bins = 20, alpha = 0.7, label='random')#, density=True)
        plt.title('{} - {}'.format(p, dist))
        plt.legend()
        new_key = 'S' + dist
        plt.subplot(122)
        plt.hist(df_same[new_key], bins = 20, alpha = 0.7, label='same SEMD')
        plt.hist(df_random[new_key], bins = 20, alpha = 0.7, label='random SEMD')
        plt.title('{} - {}'.format(p, new_key))
        plt.legend()
        plt.savefig(cwd + '/uncertainty/1D/plots/1duncertainty_norm_emd_semd_{}_{}'.format(p, dist))
        plt.show()
        #print(df_same[new_key].head())
        #print(df_random[new_key].head())

emd_keys = ('EMD_x', 'EMD_y', 'EMD_t', 'EMD_xy', 'EMD_xt', 'EMD_yt')
semd_keys = ('SEMD_x', 'SEMD_y', 'SEMD_t', 'SEMD_xy', 'SEMD_xt', 'SEMD_yt')
photons = np.arange(100, 2600, 200)
for p in photons:
    df_same = pd.read_hdf(cwd + '/uncertainty/1D/same/1duncertainty_norm_emd_semd_{}'.format(p))
    df_random = pd.read_hdf(cwd + '/uncertainty/1D/random_events/1duncertainty_norm_emd_semd_random_{}'.format(p))
    print(p)
    for dist in emd_keys:
        plt.figure(figsize = (20,5))
        plt.subplot(121)
        plt.hist(df_same[dist], bins = 20, alpha = 0.7, label='same')#, density=True)
        plt.title('{} - {}'.format(p, dist))
        plt.legend()
        new_key = 'S' + dist
        plt.subplot(122)
        plt.hist(df_same[new_key], bins = 20, alpha = 0.7, label='same SEMD')
        plt.title('{} - {}'.format(p, new_key))
        plt.legend()
        plt.savefig(cwd + '/uncertainty/1D/plots/1duncertainty_norm_onlysame_emd_semd_{}_{}'.format(p, dist))
        plt.show()
        #print(df_same[new_key].head())
        #print(df_random[new_key].head())

mean_df = pd.DataFrame(columns = ['EMD_x', 'EMD_y', 'EMD_t', 'EMD_xy', 'EMD_xt', 'EMD_yt',
                                    'SEMD_x', 'SEMD_y', 'SEMD_t', 'SEMD_xy', 'SEMD_xt', 'SEMD_yt'])
mean_rand = pd.DataFrame(columns = ['EMD_x', 'EMD_y', 'EMD_t', 'EMD_xy', 'EMD_xt', 'EMD_yt',
                                    'SEMD_x', 'SEMD_y', 'SEMD_t', 'SEMD_xy', 'SEMD_xt', 'SEMD_yt'])
for dist in emd_keys:
    help_arr_same = []
    help_arr_rand = []
    for p in photons:
        df_same = pd.read_hdf(cwd + '/uncertainty/1D/same/1duncertainty_norm_emd_semd_{}'.format(p))
        df_random = pd.read_hdf(cwd + '/uncertainty/1D/random_events/1duncertainty_norm_emd_semd_random_{}'.format(p))
        help_arr_rand.append(df_random[dist].median(axis=0))
        help_arr_same.append(df_same[dist].median(axis=0))
    mean_df[dist] = help_arr_same
    mean_rand[dist] = help_arr_rand
for dist in semd_keys:
    help_arr_same = []
    help_arr_rand = []
    for p in photons:
        df_same = pd.read_hdf(cwd + '/uncertainty/1D/same/1duncertainty_norm_emd_semd_{}'.format(p))
        df_random = pd.read_hdf(cwd + '/uncertainty/1D/random_events/1duncertainty_norm_emd_semd_random_{}'.format(p))
        help_arr_rand.append(df_random[dist].median(axis=0))
        help_arr_same.append(df_same[dist].median(axis=0))
    mean_df[dist] = help_arr_same
    mean_rand[dist] = help_arr_rand

x = np.arange(200, 5000, 200)
all_keys = ('EMD_x', 'EMD_y', 'EMD_t', 'EMD_xy', 'EMD_xt', 'EMD_yt', 'SEMD_x', 'SEMD_y', 'SEMD_t', 'SEMD_xy', 'SEMD_xt', 'SEMD_yt')
for dist in all_keys:
    plt.plot(photons, mean_df[dist], '.b', label = 'same')
    plt.plot(photons, mean_rand[dist], '.r', label = 'rand')
    plt.title('{}'.format(dist))
    plt.legend()
    plt.savefig(cwd + '/uncertainty/1D/plots/1duncertainty_norm_emd_semd_mean_{}'.format(dist))
    plt.show()

for dist in all_keys:
    plt.plot(photons, mean_df[dist], '.b', label = 'same')
    plt.title('{}'.format(dist))
    plt.legend()
    plt.savefig(cwd + '/uncertainty/1D/plots/1duncertainty_norm_emd_semd_mean_onlysame_{}'.format(dist))
    plt.show()
