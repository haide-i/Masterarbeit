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

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import itertools
import sys
sys.path.append('home/ihaide/Masterarbeit/metrics_test/photon_data/OpticalGun/')
from eulerwinkel import get_momenta, get_angle

datadir = 'diffpoints2'
ceph = f'/ceph/ihaide/ogun/Gauss/grid/{datadir}/'

pos_df = pd.read_hdf(ceph + 'distance/ogun_positiongrid_diffpoints_distall.h5')
mom_df = pd.read_hdf(ceph + 'distance/ogun_momentumgrid_diffpoints_distall.h5')
startpoint = np.loadtxt(ceph + 'startpoints.txt').round(4)
#pos_small_df = pd.read_hdf(ceph + 'ogun_positiongrid_distall_smallphrange.h5')
#mom_small_df = pd.read_hdf(ceph + 'ogun_momentumgrid_distall_smallphrange.h5')

photons = pos_df.nr_photons.unique()
comb = ('z', 'x', 'y')
groundcoord = {'x' : startpoint[0], 
                'y' : startpoint[1],
                'z' : startpoint[2]}
for var in list(itertools.combinations(comb, 2)):
    novar = list(set(comb) - set(var))
    for p in photons:
        help_df = pos_df[pos_df.nr_photons == p]
        help_df = help_df[help_df.round(4).comp_y <= 0.8]
        help_df = help_df[help_df.round(4)[f'comp_{novar[0]}'] == groundcoord[novar[0]]]
        bins = {'x' : len(help_df.round(4).drop_duplicates(subset=['comp_x'])), 
                'y' : len(help_df.round(4).drop_duplicates(subset=['comp_y'])),
                'z' : len(help_df.round(4).drop_duplicates(subset=['comp_z']))}
        length = help_df[f'comp_{var[0]}'].max(axis = 0) - help_df[f'comp_{var[0]}'].min(axis = 0)
        height = help_df[f'comp_{var[1]}'].max(axis = 0) - help_df[f'comp_{var[1]}'].min(axis = 0)
        if var[1] == 'x':
            plt.figure(figsize = (length + 2, height))
        if var[1] =='y':
            plt.figure(figsize = (length, 2*height))
        plt.hist2d(help_df[f'comp_{var[0]}'].round(4), help_df[f'comp_{var[1]}'].round(4), bins=(bins[var[0]], bins[var[1]]), cmin = 0.00001, cmap = 'YlGnBu',  weights = help_df.KS_dist, norm=mcolors.PowerNorm(0.7))
        plt.xlabel(f'{var[0]}')
        plt.ylabel(f'{var[1]}')
        plt.title(f'{int(p)} Photons - {var[0]} - {var[1]} plane')
        plt.colorbar()
        plt.savefig(f'./plots/pos_mom_dis/ks_dist_overdist_{var[0]}_{var[1]}_lastvar0_plane_{int(p)}photons_{datadir}.pdf')
        plt.show()

momentum_arr = mom_df[['comp_px', 'comp_py', 'comp_pz']].to_numpy().T
theta, psi = get_angle(momentum_arr)
mom_df['theta'] = theta
mom_df['psi'] = psi

mom_df[mom_df.KS_dist < 0.3]

n = 3
for p in photons:
    help_df = mom_df[mom_df.nr_photons == p]
    help_df = help_df.round(n).drop_duplicates(subset = ['theta', 'psi'])
    bintheta = np.sort(help_df.round(n).theta.unique()) + 0.05
    binpsi = np.sort(help_df.round(n).psi.unique()) + 0.05
    plt.hist2d(help_df.theta, help_df.psi, cmap = 'YlGnBu', cmin = 0.01, bins = (bintheta, binpsi), weights = help_df.KS_dist, norm=mcolors.PowerNorm(0.3))
    plt.xlabel('theta')
    plt.ylabel(f'psi')
    plt.title(f'{int(p)} Photons - theta - psi plane')
    plt.colorbar(label = 'KS distance')
    plt.savefig(f'./plots/pos_mom_dis/ks_dist_overdist_theta_psi_plane_{datadir}_{int(p)}photons.pdf')
    plt.show()

photons = (10., 50., 150.)
psiplot = mom_df.round(4).psi.unique()
for p in photons:
    plt.figure(figsize = (20, 10))
    theta0 = 328.9675
    thetaplus = np.round(theta0 + 0.1, 4)
    thetaminus = np.round(theta0 - 0.1, 4)
    help_df = mom_df[mom_df.nr_photons == p]
    theta1deg = help_df[help_df.round(4).theta == thetaplus].sort_values(by = 'psi')
    thetam1deg = help_df[help_df.round(4).theta == thetaminus].sort_values(by = 'psi')
    thetaex = help_df[help_df.round(4).theta == theta0].sort_values(by = 'psi')
    plt.plot(theta1deg.psi, theta1deg.KS_dist, '--', label = '1 degree plus')
    plt.plot(thetam1deg.psi, thetam1deg.KS_dist, '-.', label = '1 degree minus')
    plt.plot(thetaex.psi, thetaex.KS_dist, ':', label = '0')
    plt.legend()
    plt.show()


