#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from basf2 import *
from ROOT import Belle2
from pandas import DataFrame
import pandas as pd
import numpy as np
import os
import sys
from enum import Enum
import os
from argparse import ArgumentParser

jobID = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
if not jobID:
    jobID = int(os.environ.get("SLURM_JOB_ID", "0"))
if not jobID:
    jobID = int(os.environ.get("SLURM_JOBID", "0"))
if not jobID:
    jobID = int(os.environ.get("LSB_JOBINDEX", "0"))

ap = ArgumentParser('')
ap.add_argument('--output', default=f'ogun_{jobID}.h5', help='output filename')
ap.add_argument('--pos-dist', default='fixed', help='position distribution of emission')
ap.add_argument('--x', default=0.0, help='x emission location')
ap.add_argument('--y', default=0.0, help='y emission location')
ap.add_argument('--z', default=0.0, help='z emission location')
ap.add_argument('--dir-dist', default='fixed', help='direction distrubtion of emission')
ap.add_argument('--phi', default=0.0, help='phi emission direction')
ap.add_argument('--theta', default=180.0, help='theta emission direction')
ap.add_argument('--psi', default=0.0, help='psi emission direction')
ap.add_argument('--polx', default=0.0, help='polarization x')
ap.add_argument('--poly', default=0.0, help='polarization y')
ap.add_argument('--polz', default=1.0, help='polarization z')
ap.add_argument('--pol-dist', default='fixed', help='direction distribution of polarization')
ap.add_argument('--wave-dist', default='fixed', help='wavelength distribution')
ap.add_argument('--wavelength', default=405.0, help='wavelength in nanometers')
ap.add_argument('--numPhotons', default=1, help='number of photons per event')
opts = ap.parse_args()
output_filename = opts.output
# example of using OpticalGun to simulate laser light source
# laser source at the mirror (inside bar) emitting toward PMT's
# ---------------------------------------------------------------

class outer_zip(object):
    def __init__(self, array1, array2):
        self.len = max(len(array1), len(array2))
        self.array1 = array1
        self.array2 = array2

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        value1 = None
        if self.i < len(self.array1):
            value1 = self.array1[self.i]
        value2 = None
        if self.i < len(self.array2):
            value2 = self.array2[self.i]
        if self.i >= self.len:
            raise StopIteration
        self.i += 1
        return value1, value2

class FindCherenkovPhotons(Module):
    def initialize(self):
        self.index = self.jobid * 100000
        self.photons = []

    def event(self):
        mcp = Belle2.PyStoreArray("MCParticles")
        digits = Belle2.PyStoreArray("TOPDigits")
        production_time = -1
        production_x = np.NaN
        production_y = np.NaN
        production_z = np.NaN
        production_px = np.NaN
        production_py = np.NaN
        production_pz = np.NaN
        production_e = np.NaN
        for m, d in outer_zip(mcp, digits):
            production_time = m.getProductionTime()
            point = m.getProductionVertex()
            production_x = point.X()
            production_y = point.Y()
            production_z = point.Z()
            momentum = m.getMomentum()
            production_px = momentum.X()
            production_py = momentum.Y()
            production_pz = momentum.Z()
            production_e = m.getEnergy()
            if d is None:
                pixel_x = -1
                pixel_y = -1
                detection_time = -1
            else:
                pixel_x = d.getPixelCol()
                pixel_y = d.getPixelRow()
                detection_time = d.getTime()
            self.photons.append(
                                (
                                 production_time,
                                 production_x,
                                 production_y,
                                 production_z,
                                 production_px,
                                 production_py,
                                 production_pz,
                                 production_e,
                                 pixel_x,
                                 pixel_y,
                                 detection_time
                                )
                               )
        self.index += 1


    def terminate(self):
        photonColNames = ("production_time",
                          "production_x", "production_y", "production_z",
                          "production_px", "production_py", "production_pz", "production_e",
                          "detection_pixel_x", "detection_pixel_y", "detection_time"
                        )
        dfphotons = DataFrame(data=self.photons, columns=photonColNames)
        store = pd.HDFStore(f'{self.fname}_{self.jobid}.h5', complevel=9, complib='blosc:lz4')
        store["photons"] = dfphotons
        store.close()

# Create path
main = create_path()

# Set number of events to generate
eventinfosetter = register_module('EventInfoSetter')
eventinfosetter.param('evtNumList', [10])
main.add_module(eventinfosetter)

# Gearbox: access to database (xml files)
gearbox = register_module('Gearbox')
main.add_module(gearbox)

# Geometry
geometry = register_module('Geometry')

#geometry.param('useDB', False)
#geometry.param('components', ['TOP'])
main.add_module(geometry)

# Histogram manager immediately after master module
histo = register_module('HistoManager')
histo.param('histoFileName', f'DQMhistograms{jobID}.root')  # File to save histograms
main.add_module(histo)

# Optical gun
opticalgun = register_module('OpticalGun')
opticalgun.param('angularDistribution', 'fixed')
opticalgun.param('alpha', 30.0)
opticalgun.param('startTime', 0.0)
opticalgun.param('pulseWidth', 0.0)
opticalgun.param('numPhotons', int(opts.numPhotons))
opticalgun.param('diameter', 0.0)
opticalgun.param('barID', 5)  # if nonzero, local (bar) frame, otherwise Belle II
opticalgun.param('positionDistribution', opts.pos_dist)
opticalgun.param('x', float(opts.x))
opticalgun.param('y', float(opts.y))
opticalgun.param('z', float(opts.z))
opticalgun.param('directionDistribution', opts.dir_dist)
opticalgun.param('phi', float(opts.phi))
opticalgun.param('theta', float(opts.theta))
opticalgun.param('psi', float(opts.psi))
opticalgun.param('pol_x', float(opts.polx))
opticalgun.param('pol_y', float(opts.poly))
opticalgun.param('pol_z', float(opts.polz))
opticalgun.param('polarizationDistribution', opts.pol_dist)
opticalgun.param('wavelengthDistribution', opts.wave_dist)
opticalgun.param('wavelength', float(opts.wavelength))
main.add_module(opticalgun)

print("pos_dist", opts.pos_dist)
print("dir_dist", opts.dir_dist)


# Simulation
simulation = register_module('FullSim')
simulation.param('PhotonFraction', 1.0)
#simulation.param('deltaChordInMagneticField', 5000.0)
#simulation.param('TrackingVerbosity', 4)
#simulation.param('trajectoryDistanceTolerance', 5000.0)
#simulation.param('StandardEM', True)
#simulation.param('EmProcessVerbosity', 2)
#simulation.param('RunEventVerbosity', 2)

main.add_module(simulation)

# TOP digitization
topdigi = register_module('TOPDigitizer')
topdigi.param('useWaveforms', False)
topdigi.param('simulateTTS', False)
topdigi.param('electronicJitter', 0.0)
topdigi.param('timeZeroJitter', 0.0)
main.add_module(topdigi)

# Output
#output = register_module('RootOutput')
#output.param('outputFileName', 'opticalGun.root')
#main.add_module(output)

# save as h5
fcp = FindCherenkovPhotons()
fcp.jobid = jobID
fcp.fname = output_filename
main.add_module(fcp)

# Show progress of processing
progress = register_module('Progress')
main.add_module(progress)

# Process events
process(main)

# Print call statistics
print(statistics)
