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
import torch.multiprocessing as mp
import pickle
import os
import sys
import torch
sys.path.append('/home/ihaide/Documents/Masterarbeit-master/metrics_test/photon_data')
from ndks import ndKS
import time
ceph = '/ceph/ihaide/'


def variables(photon, evt1, evt2):
    event_sample1 = np.random.choice(np.arange(0, len(evt1)), photon)
    event_sample2 = np.random.choice(np.arange(0, len(evt2)), photon)
    xmin = -22.5
    xmax = 22.5
    ymin = -4.5
    ymax = 1.0
    tmin = 0
    tmax = 70
    x1 = torch.from_numpy((evt1.iloc[event_sample1,:].detection_pixel_x.to_numpy() + xmax)/(-xmin+xmax))
    y1 = torch.from_numpy((evt1.iloc[event_sample1,:].detection_pixel_y.to_numpy() + abs(ymin))/(ymax-ymin))
    x2 = torch.from_numpy((evt2.iloc[event_sample2,:].detection_pixel_x.to_numpy() + xmax)/(-xmin+xmax))
    y2 = torch.from_numpy((evt2.iloc[event_sample2,:].detection_pixel_y.to_numpy() + abs(ymin))/(ymax-ymin))
    t1 = torch.from_numpy((evt1.iloc[event_sample1,:].detection_time.to_numpy())/(tmin+tmax))
    t2 = torch.from_numpy((evt2.iloc[event_sample2,:].detection_time.to_numpy())/(tmin+tmax))
    return (x1, y1, t1, x2, y2, t2)


def run_distance(first_frame):
    cls = ndKS()
    last_frame = first_frame + 200
    photons = np.arange(10, 160, 10)
    keys = ['evt_idx', 'file', 'rand_evt_idx', 'rand_file', 'KS_3D']
    for p in photons:
        print(p)
        data_array = []
        ph_df = pd.DataFrame(columns = keys)
        for frame in range(first_frame, last_frame):
            start = time.time()
            file = ceph + 'photons/70ns/clean_photons_100x1E5_70ns_{}.h5'.format(frame)
            if os.path.isfile(file):
                new_df = pd.read_hdf(file)
                for event in evt_dict[frame]:
                    evt2 = new_df[new_df.evt_idx == event]
                    x1, y1, t1, x2, y2, t2 = variables(p, ground_evt, evt2)
                    data_array.append((ground_idx, 1, event, frame, cls(torch.stack((x1, y1, t1), axis=-1), torch.stack((x2, y2, t2), axis=-1))))
            end = time.time()
            print(frame, ' : ', end - start)
        data_array = np.asarray(data_array).T
        for idx, key in enumerate(keys):
            ph_df[key] = data_array[idx]
        ph_df.to_hdf('/ceph/ihaide/distances/3D/3dKS_singletoall2_70ns_{}_{}_photons{}.h5'.format(first_frame, last_frame, p), key = 'ph_df', mode = 'w', complevel=9, complib='blosc:lz4')


if __name__ == '__main__':
    a_file = open("/ceph/ihaide/distances/events_sorted.pkl", "rb")
    evt_dict = pickle.load(a_file)
    df = pd.read_hdf(ceph + 'photons/70ns/clean_photons_100x1E5_70ns_1.h5')
    ground_idx = evt_dict[1][2]
    ground_evt = df[df.evt_idx == ground_idx]
    #run_distance(0)
    processes = []
    for first in np.arange(0, 1000, 200):
        p = mp.Process(target=run_distance, args=(first,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
