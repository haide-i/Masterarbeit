import torch
import numpy as np
import os
import pandas as pd
import random
import time
import sys
sys.path.append('/home/ihaide/Documents/Masterarbeit-master/metrics_test/photon_data')
from ndks import ndKS
import pickle
#from scipy.stats import wasserstein_distance
dirname = '/ceph/ihaide/photons/first_200/'
torch.set_num_threads(5)

def variables(event, r_event, ph_length):
    draw_rand = np.arange(0, len(event))
    rand_length = np.arange(0, len(r_event))
    event_sample1 = np.random.choice(draw_rand, ph_length)
    event_samplerand = np.random.choice(rand_length, ph_length)
    event_sample2 = np.random.choice(draw_rand, ph_length)
    x1 = torch.from_numpy(np.asarray(chosen_event.iloc[event_sample1,:].detection_pixel_x.to_numpy() + 23)/46.)
    y1 = torch.from_numpy((chosen_event.iloc[event_sample1,:].detection_pixel_y.to_numpy() + 4.2)/(5.21))
    x2 = torch.from_numpy((chosen_event.iloc[event_sample2,:].detection_pixel_x.to_numpy() + 23)/46.)
    y2 = torch.from_numpy((chosen_event.iloc[event_sample2,:].detection_pixel_y.to_numpy() + 4.2)/(5.21))
    xrand = torch.from_numpy((rand_event.iloc[event_samplerand,:].detection_pixel_x.to_numpy() + 23)/46.)
    yrand = torch.from_numpy((rand_event.iloc[event_samplerand,:].detection_pixel_y.to_numpy() + 4.2)/(5.21))
    if (chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0)) != 0:
        t1 = torch.from_numpy((chosen_event.iloc[event_sample1,:].detection_time.to_numpy())/(chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0)))
        t2 = torch.from_numpy(chosen_event.iloc[event_sample2,:].detection_time.to_numpy()/(chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0)))
    else:
        t1 = torch.from_numpy((chosen_event.iloc[event_sample1,:].detection_time.to_numpy())/(chosen_event.detection_time.mean(axis=0)))
        t2 = torch.from_numpy(chosen_event.iloc[event_sample2,:].detection_time.to_numpy()/(chosen_event.detection_time.mean(axis=0)))
    if (r_event.detection_time.max(axis=0) - r_event.detection_time.min(axis=0)) != 0:
        trand = torch.from_numpy(rand_event.iloc[event_samplerand,:].detection_time.to_numpy()/(r_event.detection_time.max(axis=0) - r_event.detection_time.min(axis=0)))
    else:
        trand = torch.from_numpy(rand_event.iloc[event_samplerand,:].detection_time.to_numpy()/(r_event.detection_time.mean(axis=0)))
    return (x1, y1, t1, x2, y2, t2, xrand, yrand, trand)
    
a_file = open("/ceph/ihaide/distances/events_max200_sorted.pkl", "rb")
evt_dict = pickle.load(a_file)
event_idx = np.loadtxt('/ceph/ihaide/distances/events_max200_choose_random.txt')

use_rand = False
cls = ndKS()
alternative = False 
photon_nr = np.arange(10, 160, 10)
datanames = np.arange(0, 1000, 1)
for photons in photon_nr:
    print(photons)
    df = pd.DataFrame(columns = ['evt_idx', 'file', 'rand_evt_idx', 'rand_file', 'KS_xyt_same', 'KS_xyt_rand'])
    KS_xyt_same = []
    KS_xyt_rand = []
    evt_idx = []
    file_nb = []
    rand_evt_idx = []
    rand_file = []
    for frame in datanames:
        start = time.time()
        filename = dirname + 'clean_photons_100x1E5_first200_{}.h5'.format(frame)
        if os.path.isfile(filename):
            file = pd.read_hdf(filename)
            for event in evt_dict[frame]:
                chosen_event = file[file.evt_idx == event]
                if len(chosen_event.index) >= photons:
                    for i in range(20):
                        choose_rand = random.choice(event_idx)
                        file_rand = pd.read_hdf(dirname + 'clean_photons_100x1E5_first200_{}.h5'.format(int(choose_rand[1])))
                        rand_event = file_rand[file_rand.evt_idx == int(choose_rand[0])]
                        if len(rand_event.index) >= photons:
                            break
                    x1, y1, t1, x2, y2, t2, xrand, yrand, trand = variables(chosen_event, rand_event, photons)
                    evt_idx.append(event)
                    KS_xyt_same.append(cls(torch.stack((x1, y1, t1), axis=-1), torch.stack((x2, y2, t2), axis=-1), alternative).numpy())
                    KS_xyt_rand.append(cls(torch.stack((x1, y1, t1), axis=-1), torch.stack((xrand, yrand, trand), axis=-1), alternative).numpy())
                    file_nb.append(frame)
                    rand_evt_idx.append(int(choose_rand[0]))
                    rand_file.append(int(choose_rand[1]))
            end = time.time()
            print(frame, ' : ', start - end)
    df['evt_idx'] = evt_idx
    df['file'] = file_nb
    df['rand_evt_idx'] = rand_evt_idx
    df['rand_file'] = rand_file
    df['KS_xyt_same'] = KS_xyt_same
    df['KS_xyt_rand'] = KS_xyt_rand
    df.to_hdf('/ceph/ihaide/distances/3D/3duncertainty_all_first200_ndimkolmogorov_{}'.format(photons), key = 'df', mode = 'w')
