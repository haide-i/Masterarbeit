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
import sys
sys.path.append('/home/Documents/Masterarbeit-master/metrics_test/photon_data/')
from ndks import ndKS
from class_distance import get_dstc
import torch
import torch.multiprocessing as mp
import pickle
import time
import os
ceph = '/ceph/ihaide/'


def run_distance(first_frame):
    cls = ndKS()
    dist = get_dstc(dim=dim)
    last_frame = first_frame + 200
    photons = np.arange(10, 160, 10)
    data_array = []
    dist_array = []
    for p in photons:
        print(p)
        ph_df = pd.DataFrame(columns = keys)
        for frame in range(first_frame, last_frame):
            start = time.time()
            file = ceph + 'photons/70ns/clean_photons_100x1E5_70ns_{}.h5'.format(frame)
            if os.path.isfile(file):
                new_df = pd.read_hdf(file)
                for event in evt_dict[frame]:
                    evt2 = new_df[new_df.evt_idx == event]
                    #frame_data = [p, ground_idx, 1, event, frame]
                    #frame_data.append(dist(ground_evt, evt2, p))
                    data_array.append((p, ground_idx, 1, event, frame))
                    dist_array.append(dist(ground_evt, evt2, p))
            end = time.time()
            print(frame, ' : ', end - start)
    dist_array_new = np.asarray(dist_array).T
    data_array_new = np.asarray(data_array).T
    all_data = np.vstack((data_array_new, dist_array_new))
    for idx, key in enumerate(keys):
        ph_df[key] = all_data[idx]
    ph_df.to_hdf('/ceph/ihaide/distances/2D/2dKS_singletoall_70ns_{}_{}_photons{}_{}.h5'.format(first_frame, last_frame, np.min(photons), np.max(photons)), key = 'ph_df', mode = 'w', complevel=9, complib='blosc:lz4')


if __name__ == '__main__':
    dim = 2
    keys = ['photons', 'evt_idx', 'file', 'rand_evt_idx', 'rand_file', 'KS_3D', 'dist_p', 'dist_x']
    a_file = open("/ceph/ihaide/distances/events_sorted.pkl", "rb")
    evt_dict = pickle.load(a_file)
    df = pd.read_hdf(ceph + 'photons/70ns/clean_photons_100x1E5_70ns_1.h5')
    ground_idx = evt_dict[1][1]
    ground_evt = df[df.evt_idx == ground_idx]
    processes = []
    for first in np.arange(0, 1000, 200):
        p = mp.Process(target=run_distance, args=(first,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
