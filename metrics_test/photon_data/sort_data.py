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
from glob import glob
home = os.getenv("HOME")

dirname = '/ceph/ihaide/photons/'
file_nbr = np.arange(0, 1000, 1)
for f in file_nbr:
    file = dirname + 'photons_100x1E5_randpos_randdir_mod5_{}.h5'.format(f)
    if os.path.isfile(file):
        print(f)
        df = pd.read_hdf(dirname + 'photons_100x1E5_randpos_randdir_mod5_{}.h5'.format(f))
        clean_photons = df.loc[df.detection_time >= 0]
        clean_photons.reset_index(drop = True, inplace=True)
        clean_photons.to_hdf(dirname+'without_non_detected/clean_photons_100x1E5_randpos_randdir_mod5_{}.h5'.format(f), key = 'df', complevel=9, complib='blosc:lz4')
