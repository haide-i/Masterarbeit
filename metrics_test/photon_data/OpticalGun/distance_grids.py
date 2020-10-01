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

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('/home/ihaide/Masterarbeit/metrics_test/photon_data')
from distance import getDist
ceph = '/ceph/ihaide/ogun/Gauss/grid/diffpoints2/'


def variables(evt1):
    x = evt1.production_x.mean(axis = 0)
    y = evt1.production_y.mean(axis = 0)
    z = evt1.production_z.mean(axis = 0)
    px = evt1.production_px.mean(axis = 0)
    py = evt1.production_py.mean(axis = 0)
    pz = evt1.production_pz.mean(axis = 0)
    return x, y, z, px, py, pz


photons = (10, 20, 30, 50, 150)
keys = ['nr_photons', 'ground_x', 'ground_y', 'ground_z', 'ground_px', 'ground_py', 'ground_pz', 
        'comp_x', 'comp_y', 'comp_z', 'comp_px', 'comp_py', 'comp_pz',
       'KS_dist']
groundcoord = np.loadtxt(ceph + 'startpoints.txt').round(4) #starting coordinates in order x, y, z, psi, theta, phi
groundfile = pd.read_hdf(ceph + 'ogun_positiongrid_xrun_z50_y9_950.h5')
ground_evt = groundfile[groundfile.round(4).production_x == groundcoord[0]]
distance = []
ph = []
var1 = []
var2 = []
save_df = pd.DataFrame(columns=keys)
dstc = getDist()
for i in range(1801):
    z_diff = 0.1*int(i) - 10.*(int(i)//100)
    y_diff = 0.1*(int(i)//100)
    file = ceph + f'ogun_positiongrid_xrun_z{int(10*z_diff)}_y{int(10*y_diff)}_{i}.h5'
    if os.path.isfile(file):
        print(i)
        df = pd.read_hdf(ceph + f'ogun_positiongrid_xrun_z{int(10*z_diff)}_y{int(10*y_diff)}_{i}.h5')
        x_values = df.production_x.unique()
        for x in x_values:
            scd_evt = df[df.production_x == x]
            for p in photons:
                if scd_evt['detection_time'].mean(axis=0) >= 0:
                    distance.append(dstc(ground_evt, scd_evt, p))
                    var1.append(variables(ground_evt))
                    var2.append(variables(scd_evt))
                    ph.append(p)
distance_new = np.asarray(distance).T
var1_new = np.asarray(var1).T
var2_new = np.asarray(var2).T
p_new = np.asarray(ph).T
all_data = np.vstack((ph, var1_new, var2_new, distance_new))
for idx, key in enumerate(keys):
    save_df[key] = all_data[idx]
save_df.to_hdf(ceph + f'distance/ogun_positiongrid_diffpoints_distall.h5', key = 'save_df', mode = 'w', complevel=9, complib='blosc:lz4')

save_df.to_hdf(ceph + f'distance/ogun_positiongrid_diffpoints_distall.h5', key = 'save_df', mode = 'w', complevel=9, complib='blosc:lz4')
