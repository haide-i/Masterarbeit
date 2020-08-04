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

df_same = pd.read_hdf(cwd + '/uncertainty/2D/2duncertainty_all_ndimkolmogorov_{}'.format(200))
df_same.head()

photons1 = np.arange(10, 200, 10)
photons = np.arange(100, 2900, 100)
photons2 = np.array((3200, 3500))
photons = np.concatenate((photons1, photons, photons2))
keys = ('KS_xy', 'KS_xt', 'KS_yt')
for p in photons:
    for dist in keys:
        df_same = pd.read_hdf(cwd + '/uncertainty/2D/2duncertainty_all_ndimkolmogorov_{}'.format(p))
        same = dist + '_same'
        rand = dist + '_rand'
        plt.hist(df_same[same].transform(lambda x: x.numpy()), bins = 20, alpha = 0.7, label='same')#, density=True)
        plt.hist(df_same[rand].transform(lambda x: x.numpy()), bins = 20, alpha = 0.7, label='random')#, density=True)
        plt.title('{} - {}'.format(p, dist))
        plt.legend()
        plt.savefig(cwd + '/uncertainty/2D/plots/2duncertainty_all_ndimkolmogorov_{}_{}'.format(p, dist))
        plt.show()

mean_df = pd.DataFrame(columns = ['KS_xy', 'KS_xt', 'KS_yt'])
mean_rand = pd.DataFrame(columns = ['KS_xy', 'KS_xt', 'KS_yt'])
for dist in keys:
    help_arr_same = []
    help_arr_rand = []
    for p in photons:
        df_same = pd.read_hdf(cwd + '/uncertainty/2D/2duncertainty_all_ndimkolmogorov_{}'.format(p))
        same = dist + '_same'
        rand = dist + '_rand'
        help_arr_same.append(df_same[same].mean(axis = 0))
        help_arr_rand.append(df_same[rand].mean(axis = 0))
    mean_df[dist] = help_arr_same
    mean_rand[dist] = help_arr_rand

for dist in keys:
    plt.plot(photons, mean_df[dist], '.b', label = 'same')
    plt.plot(photons, mean_rand[dist], '.r', label = 'rand')
    plt.title('{}'.format(dist))
    plt.legend()
    plt.savefig(cwd + '/uncertainty/2D/plots/2duncertainty_mean_ndimkolmogorov_{}'.format(dist))
    plt.show()
mean_df.head()
