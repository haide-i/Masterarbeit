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
import time
import faulthandler; faulthandler.enable()
import multiprocessing as mp
ceph = '/ceph/ihaide/'


def compute_dist(evt1, evt2, p=True):
    if p:
        p1 = np.asarray((evt1.production_px.mean(axis=0), evt1.production_py.mean(axis=0), evt1.production_pz.mean(axis=0)))
        p2 = np.asarray((evt2.production_px.mean(axis=0), evt2.production_py.mean(axis=0), evt2.production_pz.mean(axis=0)))
        return np.sum((p1 - p2)**2)
    else:
        x1 = np.asarray((evt1.production_x.mean(axis=0), evt1.production_y.mean(axis=0), evt1.production_z.mean(axis=0)))
        x2 = np.asarray((evt2.production_x.mean(axis=0), evt2.production_y.mean(axis=0), evt2.production_z.mean(axis=0)))
        return np.sum((x1 - x2)**2)


def cmp_phase_dist(first_frame):
    diff = []
    for frame in range(first_frame, first_frame+100):
        start = time.time()
        file = ceph + 'photons/without_non_detected/clean_photons_100x1E5_randpos_randdir_mod5_{}.h5'.format(frame)
        if os.path.isfile(file):
            new_df = pd.read_hdf(file)
            for event in evt_dict[frame]:
                evt2 = new_df[new_df.evt_idx == event]
                diff.append((compute_dist(ground_evt, evt2, p=True), compute_dist(ground_evt, evt2, p=False)))
        end = time.time()
        print(frame, ' : ', end - start)
    np.savetxt('./phasespace_dist_groundevt1_0_{}.txt'.format(first_frame), np.asarray(diff))


if __name__ == '__main__':
    a_file = open("/ceph/ihaide/distances/events_sorted.pkl", "rb")
    evt_dict = pickle.load(a_file)

    photons = np.arange(10, 160, 10)
    df = pd.read_hdf(ceph + 'photons/70ns/clean_photons_100x1E5_70ns_1.h5')
    ground_idx = evt_dict[1][0]
    print(ground_idx)
    ground_evt = df[df.evt_idx == ground_idx]
    print(ground_evt.head())
    df.head()
    #for first in np.arange(0, 1000, 100):
    #    cmp_phase_dist(first)

df[df.evt_idx == 200].head()

# +
frames = np.arange(100, 1000, 100)
diff_all = np.loadtxt('./phasespace_dist_groundevt1_1_0.txt')

for i in frames:
    diff_all = np.vstack((diff_all, np.loadtxt('./phasespace_dist_groundevt1_1_{}.txt'.format(i))))
diff_all = diff_all.T
# -

photons = np.arange(10, 160, 10)
for p in photons:
    df = pd.read_hdf(ceph + 'distances/3D/3dKS_singletoall_70ns_photons{}.h5'.format(p))
    df['dist_p'] = diff_all[0]
    df['dist_x'] = diff_all[1]
    df.to_hdf(ceph + 'distances/3D/3dKS_singletoall_70ns_withphasediff_photons{}.h5'.format(p), key = 'df', mode = 'w', complevel=9, complib='blosc:lz4')
