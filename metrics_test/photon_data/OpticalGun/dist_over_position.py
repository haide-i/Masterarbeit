import numpy as np
import pandas as pd
import os
import sys
sys.path.append('/home/ihaide/Masterarbeit/metrics_test/photon_data')
from class_distance import get_dstc
ceph = '/ceph/ihaide/ogun/Gauss/'


def variables(evt1):
    x = evt1.production_x.mean(axis=0)
    y = evt1.production_y.mean(axis=0)
    z = evt1.production_z.mean(axis=0)
    px = evt1.production_px.mean(axis=0)
    py = evt1.production_py.mean(axis=0)
    pz = evt1.production_pz.mean(axis=0)
    return x, y, z, px, py, pz

var = 'x'
sigma_name = np.arange(0, 100, 2)
mu_diff = np.arange(0, 50, 5)
dstc = get_dstc()
photon_nr = (50, 150)
keys = ['Sigma', 'Mu', 'production_x', 'production_y', 'production_z', 
    'production_px', 'production_py', 'production_pz', 'KS_distance', 
    'momentum_distance', 'position_distance']
distance_df = pd.DataFrame(columns=keys)

ground_file = ceph + f'ogun_gauss_{var}_mu0_sigma0_1.h5'
ground_evt = pd.read_hdf(ground_file)
for p in photon_nr:
    distance = []
    mu_sigma = []
    no_exist = []
    not_detected = []
    for sigma in sigma_name:
        for mu in mu_diff:
            if sigma == 0 and mu == 0:
                groundfile_nr = 1
                add = 18
            else:
                groundfile_nr = sigma*100 + mu*4
                add = 19
            print('sigma: ', sigma, ' mu: ', mu, ' file: ', groundfile_nr)
            end_file = ceph + f'ogun_gauss_{var}_mu{mu}_sigma{sigma}_{groundfile_nr + add}.h5'
            if os.path.isfile(end_file):
                if ground_evt.detection_time.mean(axis=0) > 0:
                    print(groundfile_nr)
                    for i in range(1, 20):
                        file_nr = sigma*100 + mu*4 + i
                        new_file = ceph + f'ogun_gauss_{var}_mu{mu}_sigma{sigma}_{file_nr}.h5'
                        if os.path.isfile(new_file):
                            evt2 = pd.read_hdf(ceph + f'ogun_gauss_{var}_mu{mu}_sigma{sigma}_{file_nr}.h5')
                            if evt2.detection_time.mean(axis=0) > 0:
                                distance.append(dstc(ground_evt, evt2, p))
                                x, y, z, px, py, pz = variables(evt2)
                                mu_sigma.append((sigma, mu, x, y, z, px, py, pz))
    distance_new = np.asarray(distance).T
    mu_sigma_new = np.asarray(mu_sigma).T
    all_data = np.vstack((mu_sigma_new, distance_new))
    for idx, key in enumerate(keys):
        distance_df[key] = all_data[idx]
    distance_df.to_hdf(ceph + f'distance/ogun_groundfl_{var}_dist_photons{p}.h5', 
                    key = 'distance_df', mode = 'w', complevel=9, complib='blosc:lz4')
