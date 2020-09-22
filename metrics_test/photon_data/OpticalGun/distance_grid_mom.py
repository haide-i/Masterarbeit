import pandas as pd
import numpy as np
import os
import sys
sys.path.append('/home/ihaide/Documents/Masterarbeit-master/metrics_test/photon_data')
from class_distance import get_dstc
ceph = '/ceph/ihaide/ogun/Gauss/grid/'
def variables(evt1):
    x = evt1.production_x.mean(axis = 0)
    y = evt1.production_y.mean(axis = 0)
    z = evt1.production_z.mean(axis = 0)
    px = evt1.production_px.mean(axis = 0)
    py = evt1.production_py.mean(axis = 0)
    pz = evt1.production_pz.mean(axis = 0)
    return x, y, z, px, py, pz

photons = (10, 20, 30, 50)
keys = ['nr_photons', 'ground_x', 'ground_y', 'ground_z', 'ground_px', 'ground_py', 'ground_pz', 
        'comp_x', 'comp_y', 'comp_z', 'comp_px', 'comp_py', 'comp_pz',
       'KS_xy', 'KS_xt', 'KS_yt']
groundfile = pd.read_hdf(ceph + 'ogun_momentumgrid_psirun_phiset_theta50_50.h5').round(4)
ground_evt = groundfile[groundfile.production_px == 0.4622]
distance = []
ph = []
var1 = []
var2 = []
save_df = pd.DataFrame(columns=keys)
dstc = get_dstc(dim=2)
for i in range(101):
    print(i)
    theta_diff = 0.1*int(i)
    df = pd.read_hdf(ceph + f'ogun_momentumgrid_psirun_phiset_theta{int(10*theta_diff)}_{i}.h5')
    x_values = df.round(4).production_px.unique()
    for x in x_values:
        scd_evt = df[df.round(4).production_px == x]
        for p in photons:
            if scd_evt['detection_time'].mean(axis=0) >= 0:
                distance.append(dstc(ground_evt, scd_evt, p))
                var1.append(variables(ground_evt))
                var2.append(variables(scd_evt))
                ph.append(p)
distance_new = np.asarray(distance).T
var1_new = np.asarray(var1).T
var2_new = np.asarray(var2).T
p_new = np.asarray(ph).T
all_data = np.vstack((ph, var1_new, var2_new, distance_new))
for idx, key in enumerate(keys):
    save_df[key] = all_data[idx]
save_df.to_hdf(ceph + f'distance/ogun_momentumgrid_2dimKS_distall_smallphrange.h5', key = 'save_df', mode = 'w', complevel=9, complib='blosc:lz4')
