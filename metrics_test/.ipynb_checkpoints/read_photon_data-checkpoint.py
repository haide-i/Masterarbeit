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

# +
home = os.getenv("HOME")
filename = home + "/data/photons/photons_100x1E5_randpos_randdir_mod5_99.h5"

with h5.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
print(data)
# -

f = h5.File(filename, 'r')
df = pd.read_hdf(filename)
df.head()
print(list(f.keys()))
dset = f['photons']
print(list(dset.keys()))
axis0 = dset['axis0']
axis0.shape
block_0 = dset["block0_items"]
print(block_0.shape)
block_0.dtype

df.head()
