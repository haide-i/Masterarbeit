from loop_ogun import randOpticalGun
import numpy as np
from argparse import ArgumentParser
import time
import faulthandler; faulthandler.enable()

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
ap.add_argument('--numPhotons', default=50000, help='number of photons per event')
opts = ap.parse_args()

sigma = 0.1*int(opts.i)
print('sigma: ', sigma)

output_filename = 'ogun_gauss_{}_muplus_sigma{}'.format(opts.var, int(10*sigma))
randOpticalGun(output_filename, int(opts.i), sigma=sigma, num_photons = opts.numPhotons, var = opts.var, x=float(opts.x), y=float(opts.y), z=float(opts.z), phi=float(opts.phi), theta=float(opts.theta), psi=float(opts.psi))


