import h5py as h5
import numpy as np
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import random
import time
import torch
filenames = '/ceph/ihaide/photons/without_non_detected/'
savename = '/ceph/ihaide/distances/1D/'

def semd(p,t):
    p = np.asarray(p)
    t = np.asarray(t)
    cdf_p = np.cumsum(p)/p.size
    cdf_t = np.cumsum(t)/t.size
    d_cdf = cdf_p - cdf_t
    d_cdf_s = np.sum(np.power(d_cdf, 2))
    return np.mean(d_cdf_s)

def semd_torch(p, t):
    p = torch.from_numpy(p)
    t = torch.from_numpy(t)
    cdf_p = torch.cumsum(p, dim = 0)
    cdf_t = torch.cumsum(t, dim = 0)
    # the difference of these two gives the argument of the SEMD norm
    d_cdf = cdf_p - cdf_t
    # square
    d_cdf_s = torch.sum(torch.pow(d_cdf, 2.0))
    # return the mean of the semd
    E_E = torch.mean(d_cdf_s)
    return E_E

def emd(u_values, v_values):
    u_values = np.asarray(u_values, dtype=float)
    v_values = np.asarray(v_values, dtype=float)
    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    all_values = np.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')

    deltas = np.diff(all_values)

    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    u_cdf = u_cdf_indices / u_values.size
    
    v_cdf = v_cdf_indices / v_values.size

    return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))

def norm(p, t=None):
    if not isinstance(t, np.ndarray):
        return p/np.sum(p)
    else:
        sqr_sum = (p*p + t*t)**0.5
        return sqr_sum/np.sum(sqr_sum)

a_file = open("/ceph/ihaide/distances/events_sorted.pkl", "rb")
evt_dict = pickle.load(a_file)
event_idx = np.loadtxt('/ceph/ihaide/distances/events_choose_random.txt')
first = input('file number: ')
first = int(first)
last = first + 10 
torch.set_num_threads(5)
datanames = np.arange(first, last, 1)
use_rand = True
photons1 = np.arange(10, 200, 10)
photons = np.arange(200, 1500, 200)
photons = np.concatenate((photons1, photons))
df_keys = ['evt_idx', 'file', 'rand_evt_idx', 'rand_file', 'x', 'y', 't', 'xy', 'xt', 'yt']
for p in photons:
    print(p)
    df_emd = pd.DataFrame(columns = df_keys)
    dfrand_emd = pd.DataFrame(columns = df_keys)
    df_semd = pd.DataFrame(columns = df_keys)
    dfrand_semd = pd.DataFrame(columns = df_keys)
    EMD = []
    SEMD = []
    rand_EMD = []
    rand_SEMD = []
    for frame in datanames:
        start = time.time()
        filename = filenames + 'clean_photons_100x1E5_randpos_randdir_mod5_{}.h5'.format(frame)
        if os.path.isfile(filename):
            file = pd.read_hdf(filenames + 'clean_photons_100x1E5_randpos_randdir_mod5_{}.h5'.format(frame))
            for event in evt_dict[frame]:
                chosen_event = file[file.evt_idx == event]
                choose_rand = random.choice(event_idx)
                choose_rand[0] = int(choose_rand[0])
                choose_rand[1] = int(choose_rand[1])
                file_rand = pd.read_hdf(filenames + 'clean_photons_100x1E5_randpos_randdir_mod5_{}.h5'.format(int(choose_rand[1])))
                rand_event = file_rand[file_rand.evt_idx == int(choose_rand[0])]
                rand_length = np.arange(0, len(rand_event))
                draw_rand = np.arange(0, len(chosen_event))
                event_sample1 = np.random.choice(draw_rand, p)
                event_samplerand = np.random.choice(rand_length, p)
                event_sample2 = np.random.choice(draw_rand, p)
                x = (np.asarray((chosen_event.iloc[event_sample1,:].detection_pixel_x, chosen_event.iloc[event_sample2,:].detection_pixel_x, rand_event.iloc[event_samplerand,:].detection_pixel_x)) + 23)/46.
                y = (np.asarray((chosen_event.iloc[event_sample1,:].detection_pixel_y, chosen_event.iloc[event_sample2,:].detection_pixel_y, rand_event.iloc[event_samplerand,:].detection_pixel_y)) + 4.2)/(4.2+1.01)
                t = np.asarray(((chosen_event.iloc[event_sample1,:].detection_time)/(chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0)), \
                                chosen_event.iloc[event_sample2,:].detection_time/(chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0)), \
                                rand_event.iloc[event_samplerand,:].detection_time/(chosen_event.detection_time.max(axis=0) - chosen_event.detection_time.min(axis=0))))
                EMD.append((event, frame, choose_rand[0], choose_rand[1], emd(norm(x[0]), norm(x[1])), emd(norm(y[0]), norm(y[1])), emd(norm(t[0]), norm(t[1])), emd(norm(x[0], x[0]), norm(y[1], y[1])), emd(norm(x[0], x[0]), norm(t[1], t[1])), emd(norm(y[0], y[0]), norm(t[1], t[1]))))
                SEMD.append((event, frame, choose_rand[0], choose_rand[1], semd_torch(norm(x[0]), norm(x[1])), semd_torch(norm(y[0]), norm(y[1])), semd_torch(norm(t[0]), norm(t[1])), semd_torch(norm(x[0], x[0]), norm(y[1], y[1])), semd_torch(norm(x[0], x[0]), norm(t[1], t[1])), semd_torch(norm(y[0], y[0]), norm(t[1], t[1]))))
                rand_EMD.append((event, frame, choose_rand[0], choose_rand[1], emd(norm(x[2]), norm(x[1])), emd(norm(y[2]), norm(y[1])), emd(norm(t[2]), norm(t[1])), emd(norm(x[2], x[2]), norm(y[1], y[1])), emd(norm(x[2], x[2]), norm(t[1], t[1])), emd(norm(y[2], y[2]), norm(t[1], t[1]))))
                rand_SEMD.append((event, frame, choose_rand[0], choose_rand[1], semd_torch(norm(x[2]), norm(x[1])), semd_torch(norm(y[2]), norm(y[1])), semd_torch(norm(t[2]), norm(t[1])), semd_torch(norm(x[2], x[2]), norm(y[1], y[1])), semd_torch(norm(x[2], x[2]), norm(t[1], t[1])), semd_torch(norm(y[2], y[2]), norm(t[1], t[1]))))
                end = time.time()
        print(frame, ' : ', end - start)
    EMD = np.asarray(EMD).T
    SEMD = np.asarray(SEMD).T
    rand_EMD = np.asarray(rand_EMD).T
    rand_SEMD = np.asarray(rand_SEMD).T
    for idx, column in enumerate(df_keys):
        df_emd[column] = EMD[idx]
        df_semd[column] = SEMD[idx]
        dfrand_emd[column] = rand_EMD[idx]
        dfrand_semd[column] = rand_SEMD[idx]
    dfrand_emd.to_hdf(savename + '1duncertainty_{}_{}_size_emd_random_{}'.format(first, last, p), key = 'dfrand_emd', mode = 'w')     
    df_emd.to_hdf(savename + '1duncertainty_{}_{}_size_emd_{}.h5'.format(first, last, p), key = 'df_emd', mode = 'w')
    dfrand_semd.to_hdf(savename + '1duncertainty_{}_{}_size_semd_random_{}.h5'.format(first, last, p), key = 'dfrand_semd', mode = 'w')     
    df_semd.to_hdf(savename + '1duncertainty_{}_{}_size_semd_{}.h5'.format(first, last, p), key = 'df_semd', mode = 'w') 
