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
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
home = os.getenv("HOME")
cwd = os.getcwd()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
from ekp_style import set_ekp_style
set_ekp_style(set_sizes=True, set_background=True, set_colors=True)

uncertainty_filenames7000 = glob(cwd + '/uncertainty/2D/2duncertainty_ndimkolmogorov_1000_*')
files7000 = [pd.read_hdf(f) for f in uncertainty_filenames7000]
uncertainty_filenames100 = glob(cwd + '/uncertainty/2D/2duncertainty_ndimkolmogorov_{}_*'.format(100))
files100 = [pd.read_hdf(f) for f in uncertainty_filenames100]
addto7000 = files7000[0]
addto100 = files100[0]
for i in range(1, len(files7000)):
    addto7000 = pd.concat([addto7000, files7000[i]], ignore_index = True)
    addto100 = pd.concat([addto100, files100[i]], ignore_index = True)
completeframe7000 = addto7000
completeframe100 = addto100
randomfiles7000 = glob(cwd + '/uncertainty/2D/2duncertainty_rand_ndimkolmogorov_1000_*')
randomfiles100 = glob(cwd + '/uncertainty/2D/2duncertainty_rand_ndimkolmogorov_100_*')
randomfile7000 = [pd.read_hdf(f) for f in randomfiles7000]
randomfile100 = [pd.read_hdf(f) for f in randomfiles100]
addto7000 = randomfile7000[0]
addto100 = randomfile100[0]
for i in range(1, len(files7000)):
    addto7000 = pd.concat([addto7000, randomfile7000[i]], ignore_index = True)
    addto100 = pd.concat([addto100, randomfile100[i]], ignore_index = True)
completerandom7000 = addto7000
completerandom100 = addto100

# +
completeframe100.KS_xy = completeframe100.KS_xy.transform(lambda x: x.numpy())
completeframe100.KS_xt = completeframe100.KS_xt.transform(lambda x: x.numpy())
completeframe100.KS_yt = completeframe100.KS_yt.transform(lambda x: x.numpy())

completeframe7000.KS_xy = completeframe7000.KS_xy.transform(lambda x: x.numpy())
completeframe7000.KS_xt = completeframe7000.KS_xt.transform(lambda x: x.numpy())
completeframe7000.KS_yt = completeframe7000.KS_yt.transform(lambda x: x.numpy())

completerandom100.KS_xy = completerandom100.KS_xy.transform(lambda x: x.numpy())
completerandom100.KS_xt = completerandom100.KS_xt.transform(lambda x: x.numpy())
completerandom100.KS_yt = completerandom100.KS_yt.transform(lambda x: x.numpy())

completerandom7000.KS_xy = completerandom7000.KS_xy.transform(lambda x: x.numpy())
completerandom7000.KS_xt = completerandom7000.KS_xt.transform(lambda x: x.numpy())
completerandom7000.KS_yt = completerandom7000.KS_yt.transform(lambda x: x.numpy())

# -

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.figure(figsize = (20,5))
plt.subplot(121)
plt.title(r"KS distance x-y - 100 photons")
plt.hist(completerandom100.KS_xy, bins = 50, alpha = 0.7, label='Random events')
plt.hist(completeframe100.KS_xy, bins = 50, alpha = 0.7, label='Same events')
plt.xlabel('KS distance')
plt.ylabel('Counts')
plt.legend()
plt.subplot(122)
plt.title(r"KS distance x-y - 1000 photons")
plt.hist(completerandom7000.KS_xy, bins = 50, alpha = 0.7, label='Random events')
plt.hist(completeframe7000.KS_xy, bins = 50, alpha = 0.7, label='Same events')#, alpha = 0.7)
plt.xlabel('KS distance')
plt.ylabel('Counts')
plt.legend()
plt.savefig('./uncertainty/2D/xyndimkolmogorov_transparent.pdf')
plt.show()
plt.figure(figsize = (20,5))
plt.subplot(121)
plt.title(r"KS distance xt 100 photons")
plt.hist(completerandom100.KS_xt, bins = 50, alpha = 0.7, label='Random events')
plt.hist(completeframe100.KS_xt, bins = 50, alpha = 0.7, label='Same events')
plt.xlabel('KS distance')
plt.ylabel('Counts')
plt.legend()
plt.subplot(122)
plt.title(r"KS distance xt - 1000 photons")
plt.hist(completerandom7000.KS_xt, bins = 50, alpha = 0.7, label='Random events')
plt.hist(completeframe7000.KS_xt, bins = 50, alpha = 0.7, label= 'Same events')#, alpha = 0.7)
plt.xlabel('KS distance')
plt.ylabel('Counts')
plt.legend()
plt.savefig('./uncertainty/2D/xtndimkolmogorov_transparent.pdf')
plt.show()
plt.figure(figsize = (20,5))
plt.subplot(121)
plt.title(r"KS distance y-t - 100 photons")
plt.hist(completerandom100.KS_yt, bins = 50, alpha = 0.7, label='Random events')
plt.hist(completeframe100.KS_yt, bins = 50, alpha = 0.7, label='Same events')
plt.xlabel('KS distance')
plt.ylabel('Counts')
plt.legend()
plt.subplot(122)
plt.title(r"KS distance y-t - 1000 photons")
plt.hist(completerandom7000.KS_yt, bins = 50, alpha = 0.7, label='Random events')
plt.hist(completeframe7000.KS_yt, bins = 50, alpha = 0.7, label = 'Same events')#, alpha = 0.7)
plt.xlabel('KS distance')
plt.ylabel('Counts')
plt.legend()
plt.savefig('./uncertainty/2D/xtndimkolmogorov_transparent.pdf')
plt.show()
