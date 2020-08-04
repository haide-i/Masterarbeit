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

photons1 = np.arange(10, 200, 10)
photons = np.arange(100, 2900, 100)
photons2 = np.array((3200, 3500))
photons = np.concatenate((photons1, photons, photons2))
keys = ('EMD')
for p in photons:
    df_same = pd.read_hdf(cwd + '/uncertainty/3D/3duncertainty_all_ndimkolmogorov_{}'.format(p))
    plt.hist(df_same['KS_xyt_same'], bins = 20, alpha = 0.7, label='same')#, density=True)
    plt.hist(df_same['KS_xyt_rand'], bins = 20, alpha = 0.7, label='random')#, density=True)
    plt.title('{}'.format(p))
    plt.legend()
    plt.savefig(cwd + '/uncertainty/3D/plots/3duncertainty_all_ndimkolmogorov_{}'.format(p))
    plt.show()

mean_same = []
mean_rand = []
for p in photons:
    df_same = pd.read_hdf(cwd + '/uncertainty/3D/3duncertainty_all_ndimkolmogorov_{}'.format(p))
    mean_same.append(df_same['KS_xyt_same'].mean(axis=0))
    mean_rand.append(df_same['KS_xyt_rand'].mean(axis=0))
plt.plot(photons, mean_same, '.b', label = 'same')
plt.plot(photons, mean_rand, '.r', label = 'rand')
plt.legend()
plt.savefig(cwd + '/uncertainty/3D/plots/3duncertainty_ndimkolmogorov_mean_10_3500')
plt.show()
