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
import torch
import os
import pickle
ceph = '/ceph/ihaide/'


def compute_dist(evt1, evt2, p=True):
    if p:
        p1 = np.asarray((evt1.production_px.mean(axis=0), evt1.production_py.mean(axis=0), evt1.production_pz.mean(axis=0)))
        p2 = np.asarray((evt2.production_px.mean(axis=0), evt2.production_py.mean(axis=0), evt2.production_pz.mean(axis=0)))
        return (p1 - p2)**2
    else:
        x1 = np.asarray((evt1.production_x.mean(axis=0), evt1.production_y.mean(axis=0), evt1.production_z.mean(axis=0)))
        x2 = np.asarray((evt2.production_x.mean(axis=0), evt2.production_y.mean(axis=0), evt2.production_z.mean(axis=0)))
        return (x1 - x2)**2


# +
a_file = open("/ceph/ihaide/distances/events_sorted.pkl", "rb")
evt_dict = pickle.load(a_file)

photons = np.arange(10, 160, 10)
df = pd.read_hdf(ceph + 'photons/70ns/clean_photons_100x1E5_70ns_1.h5')
ground_idx = evt_dict[1][1]
ground_evt = df[df.evt_idx == ground_idx]
diff = []

for frame in range(0, 1000):
    print(frame)
    file = ceph + 'photons/70ns/clean_photons_100x1E5_70ns_{}.h5'.format(frame)
    if os.path.isfile(file):
        new_df = pd.read_hdf(file)
        for event in evt_dict[frame]:
            evt2 = new_df[new_df.evt_idx == event]
            diff.append((compute_dist(ground_evt, evt2, p=True), compute_dist(ground_evt, evt2, p=False)))
# -

diff_new = np.asarray(diff.T)
for p in photons:
    df = pd.read_hdf(ceph + 'distances/3D/3dKS_singletoall_70ns_photons{}.h5'.format(p))
    df['dist_p'] = diff_new[0]
    df['dist_x'] = diff_new[1]
    df.to_hdf(ceph + 'distances/3D/3dKS_singletoall_70ns_withdist_photons{}.h5'.format(p), key = 'df', mode = 'w', complevel=9, complib='blosc:lz4')
