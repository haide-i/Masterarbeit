from class_opticalGun import randOpticalGun
import numpy as np
from argparse import ArgumentParser
import time

ap = ArgumentParser('')
ap.add_argument('--i', default=1, help='loop number')
ap.add_argument('--var', default='x', help='variable to opts.vary')
ap.add_argument('--output', default='test', help='output filename')
ap.add_argument('--x', default=5.0, help='x emission location')
ap.add_argument('--y', default=0.0, help='y emission location')
ap.add_argument('--z', default=0.0, help='z emission location')
ap.add_argument('--phi', default=0.0, help='phi emission direction')
ap.add_argument('--theta', default=180.0, help='theta emission direction')
ap.add_argument('--psi', default=0.0, help='psi emission direction')
ap.add_argument('--numPhotons', default=100, help='number of photons per event')
opts = ap.parse_args()

exp_number = int(opts.i)//20.
sigma = exp_number // 10. * 0.2
mu_diff = exp_number % 10 * 0.5
print('sigma: ', sigma, ' mu: ', mu_diff)
if opts.var == 'x':
    print(opts.var)
    if mu_diff < 2.5:
        opts.x = np.random.normal(float(opts.x)+mu_diff, sigma)
    else:
        opts.x = np.random.normal(float(opts.x)+2.-mu_diff, sigma)
elif opts.var == 'y':
    print(opts.var)
    if mu_diff < 2.5:
        opts.y = np.random.normal(float(opts.y)+mu_diff, sigma)
    else:
        opts.y = np.random.normal(float(opts.y)+2.-mu_diff, sigma)
elif opts.var == 'z':
    print(opts.var)
    if mu_diff < 2.5:
        opts.z = np.random.normal(float(opts.z)+mu_diff, sigma)
    else:
        opts.z = np.random.normal(float(opts.z)+2.-mu_diff, sigma)
elif opts.var == 'phi':
    print(opts.var)
    if mu_diff < 2.5:
        opts.phi = np.random.normal(float(opts.phi)+mu_diff, sigma)
    else:
        opts.phi = np.random.normal(float(opts.phi)+2.-mu_diff, sigma)
elif opts.var == 'theta':
    print(opts.var)
    if mu_diff < 2.5:
        opts.theta = np.random.normal(float(opts.theta)+mu_diff, sigma)
    else:
        opts.theta = np.random.normal(float(opts.theta)+2.-mu_diff, sigma)
elif opts.var == 'psi':
    print(opts.var)
    if mu_diff < 2.5:
        opts.psi = np.random.normal(float(opts.psi)+mu_diff, sigma)
    else:
        opts.psi = np.random.normal(float(opts.psi)+2.-mu_diff, sigma)

output_filename = 'ogun_gauss_{}_mu{}_sigma{}'.format(opts.var, int(10*mu_diff), int(10*sigma))
randOpticalGun(output_filename, int(opts.i), num_photons = opts.numPhotons, x=opts.x, y=opts.y, z=opts.z, phi=opts.phi, theta=opts.theta, psi=opts.psi)


