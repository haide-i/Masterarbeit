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
import torch.multiprocessing as mp
#from scipy.stats import wasserstein_distance
dirname = '/ceph/ihaide/photons/70ns/'

def variables(chosen_event, rand_event, ph_length):
    draw_rand = np.arange(0, len(chosen_event))
    rand_length = np.arange(0, len(rand_event))
    event_sample1 = np.random.choice(draw_rand, ph_length)
    event_samplerand = np.random.choice(rand_length, ph_length)
    event_sample2 = np.random.choice(draw_rand, ph_length)
    x1 = torch.from_numpy((chosen_event.iloc[event_sample1,:].detection_pixel_x.to_numpy() + 23)/46.)
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
    if (rand_event.detection_time.max(axis=0) - rand_event.detection_time.min(axis=0)) != 0:
        trand = torch.from_numpy(rand_event.iloc[event_samplerand,:].detection_time.to_numpy()/(rand_event.detection_time.max(axis=0) - rand_event.detection_time.min(axis=0)))
    else:
        trand = torch.from_numpy(rand_event.iloc[event_samplerand,:].detection_time.to_numpy()/(rand_event.detection_time.mean(axis=0)))
    return (x1, y1, t1, x2, y2, t2, xrand, yrand, trand)


# +
def run_distance(first):
    cls = ndKS() 
    photon_nr = np.arange(10, 200, 10)
    datanames = np.arange(first, first+20, 1)
    keys = ['evt_idx', 'file', 'rand_evt_idx', 'rand_file', 'KS_xyt_same', 'KS_xyt_rand']
    data_array = []
    for photons in photon_nr:
        print(photons)
        df = pd.DataFrame(columns = keys)
        for frame in datanames:
            start = time.time()
            filename = dirname + 'clean_photons_100x1E5_70ns_{}.h5'.format(frame)
            if os.path.isfile(filename):
                file = pd.read_hdf(filename)
                for event in evt_dict[frame]:
                    chosen_event = file[file.evt_idx == event]
                    choose_rand = random.choice(event_idx)
                    file_rand = pd.read_hdf(dirname + 'clean_photons_100x1E5_70ns_{}.h5'.format(int(choose_rand[1])))
                    rand_event = file_rand[file_rand.evt_idx == int(choose_rand[0])]
                    x1, y1, t1, x2, y2, t2, xrand, yrand, trand = variables(chosen_event, rand_event, photons)
                    data_array.append((event, frame, int(choose_rand[0]), int(choose_rand[1]), 
                                        cls(torch.stack((x1, y1, t1), axis=-1), torch.stack((x2, y2, t2), axis=-1)), 
                                        cls(torch.stack((x1, y1, t1), axis=-1), torch.stack((xrand, yrand, trand), axis=-1))))
                end = time.time()
                print(frame, ' : ', start - end)
    data_array = np.asarray(data_array).T
    for idx, key in enumerate(keys):
        df[key] = data_array[idx]
    df.to_hdf('/ceph/ihaide/distances/3D/3dKS_70ns_{}_{}_photons{}_{}.h5'.format(first, first+20, np.min(photons), np.max(photons)), 
            key = 'df', mode = 'w', complevel=9, complib='blosc:lz4')
    
if __name__ == '__main__':
	a_file = open("/ceph/ihaide/distances/events_sorted.pkl", "rb")
	evt_dict = pickle.load(a_file)
	event_idx = np.loadtxt('/ceph/ihaide/distances/events_choose_random.txt')

	processes = []
	for first in np.arange(0, 100, 20):
		p = mp.Process(target=run_distance, args=(first,))
		p.start()
		processes.append(p)
	for p in processes:
		p.join()
