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
import torch.multiprocessing as mp
import faulthandler; faulthandler.enable()
filenames = '/ceph/ihaide/photons/70ns/'
savename = '/ceph/ihaide/distances/1D/'
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
    x1 = (evt1.iloc[event_sample1,:].detection_pixel_x.to_numpy() + xmax)/(-xmin+xmax)
    y1 = (evt1.iloc[event_sample1,:].detection_pixel_y.to_numpy() + abs(ymin))/(ymax-ymin)
    x2 = (evt2.iloc[event_sample2,:].detection_pixel_x.to_numpy() + xmax)/(-xmin+xmax)
    y2 = (evt2.iloc[event_sample2,:].detection_pixel_y.to_numpy() + abs(ymin))/(ymax-ymin)
    t1 = (evt1.iloc[event_sample1,:].detection_time.to_numpy())/(tmin+tmax)
    t2 = (evt2.iloc[event_sample2,:].detection_time.to_numpy())/(tmin+tmax)
    return (x1, y1, t1, x2, y2, t2)


def semd(p,t):
    p = np.asarray(p)
    t = np.asarray(t)
    cdf_p = np.cumsum(p)
    cdf_t = np.cumsum(t)
    d_cdf = cdf_p - cdf_t
    d_cdf_s = np.sum(np.power(d_cdf, 2))
    return np.mean(d_cdf_s)

def semd_torch(p, t):
    p = torch.from_numpy(p)
    t = torch.from_numpy(t)
    p = torch.histc(p, bins=100).float().unsqueeze(0) / float(len(p))
    t = torch.histc(t, bins=100).float().unsqueeze(0) / float(len(t))
    
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

    u_values, _ = np.histogram(u_values, bins=100)#.unsqueeze(0) / float(len(u_values))
    v_values, _ = np.histogram(v_values, bins=100)#.unsqueeze(0) / float(len(v_values))
    u_values = u_values / float(len(u_values))
    v_values = v_values / float(len(v_values))
    
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

def run_distance(first):
    last = first + 200
    datanames = np.arange(first, last, 1)
    photons = np.arange(70, 160, 10)
    #photons = np.concatenate((photons1, photons))
    df_keys = ['evt_idx', 'file', 'rand_evt_idx', 'rand_file', 
               'x', 'y', 't', 'xy', 'xt', 'yt']
    for p in photons:
        print(p)
        df_emd = pd.DataFrame(columns = df_keys)
        df_semd = pd.DataFrame(columns = df_keys)
        EMD = []
        SEMD = []
        for frame in datanames:
            start = time.time()
            filename = filenames + 'clean_photons_100x1E5_70ns_{}.h5'.format(frame)
            if os.path.isfile(filename):
                file = pd.read_hdf(filename)
                for event in evt_dict[frame]:
                    ground_evt = file[file.evt_idx == event]
                    evt2 = file[file.evt_idx == event]
                    x1, y1, t1, x2, y2, t2 = variables(p, ground_evt, evt2)
                    EMD.append((event, frame, event, frame, 
                                emd(x1, x2), emd(y1, y2), emd(t1, t2), 
                                emd(np.sqrt(x1**2+y1**2)/np.sqrt(2), np.sqrt(x2**2+y2**2)/np.sqrt(2)), 
                                emd(np.sqrt(x1**2+t1**2)/np.sqrt(2), np.sqrt(x2**2+t2**2)/np.sqrt(2)), 
                                emd(np.sqrt(y1**2+t1**2)/np.sqrt(2), np.sqrt(y2**2+t2**2)/np.sqrt(2))))
                    SEMD.append((event, frame, event, frame, 
                                 semd_torch(x1, x2), semd_torch(y1, y2), semd_torch(t1, t2), 
                                 semd_torch(np.sqrt(x1**2+y1**2)/np.sqrt(2), np.sqrt(x2**2+y2**2)/np.sqrt(2)),
                                 semd_torch(np.sqrt(x1**2+t1**2)/np.sqrt(2), np.sqrt(x2**2+t2**2)/np.sqrt(2)), 
                                 semd_torch(np.sqrt(y1**2+t1**2)/np.sqrt(2), np.sqrt(y2**2+t2**2)/np.sqrt(2))))
                    end = time.time()
            print(frame, ' : ', end - start)
        EMD = np.asarray(EMD).T
        SEMD = np.asarray(SEMD).T
        for idx, column in enumerate(df_keys):
            df_emd[column] = EMD[idx]
            df_semd[column] = SEMD[idx]
        df_emd.to_hdf(savename + '1duncertainty_{}_{}_histnorm_allsame_emd_{}.h5'.format(first, last, p), key = 'df_emd', mode = 'w')
        df_semd.to_hdf(savename + '1duncertainty_{}_{}_histnorm_allsame_semd_{}.h5'.format(first, last, p), key = 'df_semd', mode = 'w')

if __name__ == '__main__':
    
    a_file = open("/ceph/ihaide/distances/events_sorted.pkl", "rb")
    evt_dict = pickle.load(a_file)
    #file_nr = 1
    #df = pd.read_hdf(ceph + 'photons/70ns/clean_photons_100x1E5_70ns_{}.h5'.format(file_nr))
    #ground_idx = evt_dict[file_nr][1]
    #ground_evt = df[df.evt_idx == ground_idx]
    ymin = -4.5
    ymax = 1.0
    xmin = -22.5
    xmax = 22.5
    tmin = 0
    tmax = 70
    #run_distance(0)
    processes = []
    for first in np.arange(0, 1000, 200):
        p = mp.Process(target=run_distance, args=(first,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

