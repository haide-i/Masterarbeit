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

ap = ArgumentParser('')
ap.add_argument('--i', default=1, help='loop number')
ap.add_argument('--var', default='x', help='variable to vary')
ap.add_argument('--output', default='test', help='output filename')
ap.add_argument('--x', default=5.0, help='x emission location')
ap.add_argument('--y', default=0.0, help='y emission location')
ap.add_argument('--z', default=0.0, help='z emission location')
ap.add_argument('--phi', default=0.0, help='phi emission direction')
ap.add_argument('--theta', default=180.0, help='theta emission direction')
ap.add_argument('--psi', default=0.0, help='psi emission direction')
ap.add_argument('--numPhotons', default=100, help='number of photons per event')
opts = ap.parse_args()
output_filename = opts.output
jobID = int(opts.i)


# example of using OpticalGun to simulate laser light source
# laser source at the mirror (inside bar) emitting toward PMT's
# ---------------------------------------------------------------

class FindCherenkovPhotons(Module):
    def initialize(self):
        self.index = self.jobid * 200
        self.photons = []

    def event(self):
        photons = Belle2.PyStoreArray("TOPSimPhotons")
        if len(photons) == 0:
            mcp = Belle2.PyStoreArray("MCParticles")
            m = mcp[0]
            #print(m)
            self.photons.append(
                                (
                                 self.index,
                                 m.getProductionTime(), #production_time,
                                 m.getVertex().x(), #production_x,
                                 m.getVertex().y(), #production_y,
                                 m.getVertex().z(), #production_z,
                                 m.getVertex().Px(), #production_px,
                                 m.getVertex().Py(), #production_py,
                                 m.getVertex().Pz(), #production_pz,
                                 m.getEnergy(), #production_e,
                                 0, # detection_x,
                                 0, # detection_y,
                                #0, # detection_z,
                                 0, # detection_px,
                                 0, # detection_py,
                                #0, # detection_pz,
                                 -1, # length,
                                 -1, # detection_time
                                )
            )
        for p in photons:
            production_time = p.getEmissionTime()
            point = p.getEmissionPoint()
            production_x = point.X()
            production_y = point.Y()
            production_z = point.Z()
            momentum = p.getEmissionDir()
            production_px = momentum.X()
            production_py = momentum.Y()
            production_pz = momentum.Z()
            production_e = p.getEnergy()
            detection_point = p.getDetectionPoint()
            detection_x = detection_point.X()
            detection_y = detection_point.Y()
            detection_z = detection_point.Z()
            # print(detection_z)
            detection_dir = p.getDetectionDir()
            detection_px = detection_dir.X()
            detection_py = detection_dir.Y()
            detection_pz = detection_dir.Z()
            length = p.getLength()
            detection_time = p.getDetectionTime()
            self.photons.append(
                                (
                                 self.index,
                                 production_time,
                                 production_x,
                                 production_y,
                                 production_z,
                                 production_px,
                                 production_py,
                                 production_pz,
                                 production_e,
                                 detection_x,
                                 detection_y,
                                #  detection_z,
                                 detection_px,
                                 detection_py,
                                #  detection_pz,
                                 length,
                                 detection_time
                                )
            )
        self.index += 1


    def terminate(self):
        photonColNames = ("evt_idx",
                          "production_time",
                          "production_x",
                          "production_y",
                          "production_z",
                          "production_px",
                          "production_py",
                          "production_pz",
                          "production_e",
                          "detection_pixel_x",
                          "detection_pixel_y",
                        #   "detection_pixel_z",
                          "detection_px",
                          "detection_py",
                        #   "detection_pz",
                          "length",
                          "detection_time"
                        )
        dfphotons = DataFrame(data=self.photons, columns=photonColNames)
        store = pd.HDFStore(f'/ceph/ihaide/ogun/Gauss/grid/{self.fname}_{self.jobid}.h5', complevel=9, complib='blosc:lz4')
        store["photons"] = dfphotons
        store.close()

z_diff = 0.1*int(opts.i) - 10.*(int(opts.i)//100)
y_diff = 0.1*(int(opts.i)//100)
output_filename = f'ogun_positiongrid_xrun_z{int(10*z_diff)}_y{int(10*y_diff)}'
wavelength = 405.0

# Create path
main = create_path()

# Set number of events to generate
eventinfosetter = register_module('EventInfoSetter')
eventinfosetter.param('evtNumList', [202])
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
# histo = register_module('HistoManager')
# histo.param('histoFileName', f'DQMhistograms{jobID}.root')  # File to save histograms
# main.add_module(histo)


# Optical gun
for i in range(101):
    x = float(opts.x)
    y = float(opts.y) + y_diff - 0.9
    z = float(opts.z) + z_diff - 5.
    phi = float(opts.phi)
    theta = float(opts.theta)
    psi = float(opts.psi)
    x = -5. + (i)*0.1 + x
    opticalgun = register_module('OpticalGun')
    opticalgun.param('angularDistribution', 'uniform')
    opticalgun.param('minAlpha', 0.0)
    opticalgun.param('maxAlpha', 0.00001)
    opticalgun.param('startTime', 0.0)
    opticalgun.param('pulseWidth', 0.0)
    opticalgun.param('numPhotons', int(opts.numPhotons))
    opticalgun.param('diameter', 0.0)
    opticalgun.param('slotID', 5)  # if nonzero, local (bar) frame, otherwise Belle II
    opticalgun.param('x', x)
    opticalgun.param('y', y)
    opticalgun.param('z', z)
    opticalgun.param('phi', phi)
    opticalgun.param('theta', theta)
    opticalgun.param('psi', psi)
    opticalgun.param('wavelength', wavelength)
    main.add_module(opticalgun)


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
# output = register_module('RootOutput')
# output.param('outputFileName', 'opticalGun.root')
# main.add_module(output)

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

