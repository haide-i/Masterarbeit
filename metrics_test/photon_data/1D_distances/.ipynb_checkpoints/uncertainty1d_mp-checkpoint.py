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
filenames = '/ceph/ihaide/photons/without_non_detected/'
savename = '/ceph/ihaide/distances/1D/'

def semd(p,t):
    p = np.asarray(p)
    t = np.asarray(t)
    cdf_p = np.cumsum(p)
    cdf_t = np.cumsum(t)
    d_cdf = cdf_p - cdf_t
    d_cdf_s = np.sum(np.power(d_cdf, 2))
    return np.mean(d_cdf_s)

def semd_torch(p, t, mnm=0, mxm=0):
    p = torch.from_numpy(p)
    t = torch.from_numpy(t)
    p = torch.histc(p, min=mnm, max=mxm, bins=100).float().unsqueeze(0) / float(len(p))
    t = torch.histc(t, min=mnm, max=mxm, bins=100).float().unsqueeze(0) / float(len(t))
    
    cdf_p = torch.cumsum(p, dim = 0)
    cdf_t = torch.cumsum(t, dim = 0)
    # the difference of these two gives the argument of the SEMD norm
    d_cdf = cdf_p - cdf_t
    # square
    d_cdf_s = torch.sum(torch.pow(d_cdf, 2.0))
    # return the mean of the semd
    E_E = torch.mean(d_cdf_s)
    return E_E

def emd(u_values, v_values, mnm=0, mxm=0):
    #if mnm is not None & mxm is not None:
    u_values, _ = np.histogram(u_values, range=(mnm, mxm), bins=100)#.unsqueeze(0) / float(len(u_values))
    v_values, _ = np.histogram(v_values, range=(mnm, mxm), bins=100)#.unsqueeze(0) / float(len(v_values))
    u_values = u_values / float(len(u_values))
    v_values = v_values / float(len(v_values))
    #else:
    #    u_values = np.histogram(u_values, bins=100).float().unsqueeze(0) / float(len(u_values))
    #    v_values = np.histogram(v_values, bins=100).float().unsqueeze(0) / float(len(v_values))
    #u_values = np.asarray(u_values, dtype=float)
    #v_values = np.asarray(v_values, dtype=float)
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
    last = first + 20
    datanames = np.arange(first, last, 1)
    use_rand = True
    photons = np.arange(10, 200, 10)
    #photons = np.concatenate((photons1, photons))
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
                file = pd.read_hdf(filename)
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
                    x = np.asarray((chosen_event.iloc[event_sample1,:].detection_pixel_x, chosen_event.iloc[event_sample2,:].detection_pixel_x, rand_event.iloc[event_samplerand,:].detection_pixel_x))
                    y = np.asarray((chosen_event.iloc[event_sample1,:].detection_pixel_y, chosen_event.iloc[event_sample2,:].detection_pixel_y, rand_event.iloc[event_samplerand,:].detection_pixel_y))
                    t = np.asarray((chosen_event.iloc[event_sample1,:].detection_time, \
                                    chosen_event.iloc[event_sample2,:].detection_time, \
                                    rand_event.iloc[event_samplerand,:].detection_time))
                    EMD.append((event, frame, choose_rand[0], choose_rand[1], emd(x[0], x[1], -23., 23.), emd(y[0], y[1], -4.2, 1.01), emd(t[0], t[1]), emd(np.sqrt(x[0]**2+y[0]**2), np.sqrt(x[1]**2+y[1]**2)), emd(np.sqrt(x[0]**2+t[0]**2), np.sqrt(x[1]**2+t[1]**2)), emd(np.sqrt(y[0]**2+t[0]**2), np.sqrt(y[1]**2+t[1]**2))))
                    SEMD.append((event, frame, choose_rand[0], choose_rand[1], semd_torch(x[0], x[1], -23., 23.), semd_torch(y[0], y[1], -4.2, 1.01), semd_torch(t[0], t[1]), semd_torch(np.sqrt(x[0]**2+y[0]**2), np.sqrt(x[1]**2+y[1]**2)), semd_torch(np.sqrt(x[0]**2+t[0]**2), np.sqrt(x[1]**2+t[1]**2)), semd_torch(np.sqrt(y[0]**2+t[0]**2), np.sqrt(y[1]**2+t[1]**2))))
                    rand_EMD.append((event, frame, choose_rand[0], choose_rand[1], emd(x[2], x[1], -23., 23.), emd(y[2], y[1], -4.2, 1.01), emd(t[2], t[1]), emd(np.sqrt(x[0]**2+y[0]**2), np.sqrt(x[2]**2+y[2]**2)), emd(np.sqrt(x[0]**2+t[0]**2), np.sqrt(x[2]**2+t[2]**2)), emd(np.sqrt(y[0]**2+t[0]**2), np.sqrt(y[2]**2+t[2]**2))))
                    rand_SEMD.append((event, frame, choose_rand[0], choose_rand[1], semd_torch(x[2], x[1], -23., 23.), semd_torch(y[2], y[1], -4.2, 1.01), semd_torch(t[2], t[1]), semd_torch(np.sqrt(x[0]**2+y[0]**2), np.sqrt(x[2]**2+y[2]**2)), semd_torch(np.sqrt(x[0]**2+t[0]**2), np.sqrt(x[2]**2+t[2]**2)), semd_torch(np.sqrt(y[0]**2+t[0]**2), np.sqrt(y[2]**2+t[2]**2))))
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
        dfrand_emd.to_hdf(savename + '1duncertainty_{}_{}_histnorm_emd_random_{}.h5'.format(first, last, p), key = 'dfrand_emd', mode = 'w')     
        df_emd.to_hdf(savename + '1duncertainty_{}_{}_histnorm_emd_{}.h5'.format(first, last, p), key = 'df_emd', mode = 'w')
        dfrand_semd.to_hdf(savename + '1duncertainty_{}_{}_histnorm_semd_random_{}.h5'.format(first, last, p), key = 'dfrand_semd', mode = 'w')     
        df_semd.to_hdf(savename + '1duncertainty_{}_{}_histnorm_semd_{}.h5'.format(first, last, p), key = 'df_semd', mode = 'w')
        #return first

if __name__ == '__main__':
    
    a_file = open("/ceph/ihaide/distances/events_sorted.pkl", "rb")
    evt_dict = pickle.load(a_file)
    event_idx = np.loadtxt('/ceph/ihaide/distances/events_choose_random.txt')
    
    #run_distance(0)
    processes = []
    for first in np.arange(0, 120, 20):
        p = mp.Process(target=run_distance, args=(first,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    #pool = mp.Pool(processes = 4)
    #first_range = np.arange(30, 70, 10)
    #results = [pool.apply(run_distance, args = (first, )) for first in first_range]
    #pool.close()
    #print(results)




