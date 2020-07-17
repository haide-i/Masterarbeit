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

dirname = home + '/data/photons/'
file_nbr = np.array([1, 2, 3, 4, 5, 6, 11, 99, 121, 264, 745])
for i in file_nbr:
    df = pd.read_hdf(dirname+'photons_100x1E5_randpos_randdir_mod5_{}.h5'.format(i))
    clean_photons = df[df.detection_time >= 0]
    clean_photons.sort_values(by=['production_x'], inplace = True)
    clean_photons.reset_index(drop = True, inplace=True)
    clean_photons.to_hdf(dirname+'without_non_det/clean_photons_100x1E5_randpos_randdir_mod5_{}.h5'.format(i), key = 'df')

check_df = pd.read_hdf(dirname + 'without_non_det/clean_photons_100x1E5_randpos_randdir_mod5_121.h5')
print(check_df.shape)
check_df.head(-1)


filenames = glob(dirname + 'without_non_det/clean_photons_100x1E5_randpos_randdir_mod5_*.h5')
dataframes = [pd.read_hdf(f) for f in filenames]
dataframes = [f.round(1) for f in dataframes]

group_names = []
for i in range(len(dataframes)):
    groups = dataframes[i].groupby(['production_x', 'production_y', 'production_z',
                                        'production_px', 'production_py', 'production_pz',
                                        'production_e'])
    print([name for name, _ in groups])
    for j in range(i+1, len(dataframes)):
        comp_group = dataframes[j].groupby(['production_x', 'production_y', 'production_z',
                                        'production_px', 'production_py', 'production_pz',
                                        'production_e'])
        for scnd_name, _ in comp_group:
            for name, _ in groups:
                if np.all(scnd_name == name):
                    print("True")
                #print(name)

# +
groups = dataframes[0].groupby(['production_x', 'production_y', 'production_z',
                                        'production_px', 'production_py', 'production_pz',
                                        'production_e'])

name = [name for name, _ in groups]
print(name)
