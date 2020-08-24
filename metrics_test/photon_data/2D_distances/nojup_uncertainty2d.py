import h5py as h5
import numpy as np
import torch
import os
import pandas as pd
import pickle
import torch.multiprocessing as mp
import time
from glob import glob
import random
import sys
sys.path.append('/home/ihaide/Documents/Masterarbeit-master/metrics_test/photon_data')
from ndks import ndKS
home = os.getenv("HOME")
filenames = '/ceph/ihaide/photons/without_non_detected/'

def variables(chosen_event, rand_event, ph_length):
    draw_rand = np.arange(0, len(chosen_event))
    rand_length = np.arange(0, len(rand_event))
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
    if (rand_event.detection_time.max(axis=0) - rand_event.detection_time.min(axis=0)) != 0:
        trand = torch.from_numpy(rand_event.iloc[event_samplerand,:].detection_time.to_numpy()/(rand_event.detection_time.max(axis=0) - rand_event.detection_time.min(axis=0)))
    else:
        trand = torch.from_numpy(rand_event.iloc[event_samplerand,:].detection_time.to_numpy()/(rand_event.detection_time.mean(axis=0)))
    return (x1, y1, t1, x2, y2, t2, xrand, yrand, trand)

def run_distance(first):
	cls = ndKS()
	alternative = False
	photons = np.arange(10, 200, 10)
	datanames = np.arange(first, first+20, 1)
	keys = ['photons', 'evt_idx', 'file', 'rand_evt_idx', 'rand_file', 'KS_xy_same', 'KS_xt_same', 'KS_yt_same', 'KS_xy_rand', 'KS_xt_rand', 'KS_yt_rand']
	df = pd.DataFrame(columns=keys)
	data_array = []
	for p in photons:
		print(p)
		for frame in datanames:
			start = time.time()
			filename = filenames + 'clean_photons_100x1E5_randpos_randdir_mod5_{}.h5'.format(frame)
			if os.path.isfile(filename):
				file = pd.read_hdf(filename)
				for event in evt_dict[frame]:
					chosen_event = file[file.evt_idx == event]
					choose_rand = random.choice(event_idx)
					file_rand = pd.read_hdf(filenames + 'clean_photons_100x1E5_randpos_randdir_mod5_{}.h5'.format(int(choose_rand[1])))
					rand_event = file_rand[file_rand.evt_idx == int(choose_rand[0])]		
					x1, y1, t1, x2, y2, t2, xrand, yrand, trand = variables(chosen_event, rand_event, p)
					data_array.append((p, event, frame, choose_rand[0], choose_rand[1], cls(torch.stack((x1, y1), axis=-1), torch.stack((x2, y2), axis=-1)), cls(torch.stack((x1, y1), axis=-1), torch.stack((x2, y2), axis=-1)), cls(torch.stack((x1, t1), axis=-1), torch.stack((x2, t2), axis=-1)), cls(torch.stack((x1, y1), axis=-1), torch.stack((xrand, yrand), axis=-1)), cls(torch.stack((x1, y1), axis=-1), torch.stack((xrand, yrand), axis=-1)), cls(torch.stack((x1, t1), axis=-1), torch.stack((xrand, trand), axis=-1))))
			end = time.time()
			print(frame, ': ', end - start)
	data_array = np.asarray(data_array).T
	for idx, key in enumerate(keys):
		df[key] = data_array[idx]
	df.to_hdf('/ceph/ihaide/distances/2D/2dKS_{}_{}_photons{}_{}.h5'.format(first, first+20, np.min(photons), np.max(photons)), key='df', mode='w', complevel=9, complib='blosc:lz4')



if __name__ == '__main__':
	a_file = open("/ceph/ihaide/distances/events_sorted.pkl", "rb")
	evt_dict = pickle.load(a_file)
	event_idx = np.loadtxt('/ceph/ihaide/distances/events_choose_random.txt')

	processes = []
	for first in np.arange(0, 120, 20):
		p = mp.Process(target=run_distance, args=(first,))
		p.start()
		processes.append(p)
	for p in processes:
		p.join()
