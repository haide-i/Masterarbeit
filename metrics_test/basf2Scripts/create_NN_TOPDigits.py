#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enum import Enum
from argparse import ArgumentParser
from basf2 import *
import ROOT
from pandas import DataFrame
import pandas as pd
from ROOT import Belle2
from tracking import add_tracking_reconstruction, add_cr_tracking_reconstruction
from svd import add_svd_reconstruction, add_svd_simulation
from pxd import add_pxd_reconstruction, add_pxd_simulation
from simulation import add_simulation
from reconstruction import add_reconstruction
import numpy as np
import os
import torch
import numpy as np
import sys
import os.path
# import model definition
sys.path.append('..')
from refl.model import mt


# command line options
ap = ArgumentParser('')
ap.add_argument('--particle', type=int, default=13, help='pdg code of the particles to generate (13, 211, 321, etc)')
# Added default in front of 'TOPOutput.root'
ap.add_argument('--output', '-o', default= 'TOPOutput.root', help='Output filename')
# Adding arguments for momentum, phi, theta, x, y, z
ap.add_argument('--momentum', type=float, default=2.0, help='what is the momentum of the particle?')
ap.add_argument('--phi', type=float, default=87.0, help='what is phi value of particle track?')
ap.add_argument('--theta', type=float, default=63.5,  help='what is theta value of particle track?')
ap.add_argument('--xVertex', type=float, default=0.0, help='what is the x vertex value?')
ap.add_argument('--yVertex', type=float, default=0.0, help='what is the y vertex value?')
ap.add_argument('--zVertex', type=float, default=0.0, help='what is the z vertex value?')
opts = ap.parse_args()

# use a pyhon3 enum for readable particle gun configuration setup
class ParticleGunConfig(Enum):
    FROMORIGIN = 0
    FROMABOVE = 1
    FROMSIDE = 2


class NN_TOPDigitMaker(Module):
    def initialize(self):
        # load the current best model
        self.model = mt.MLP(latent_depth=50, latent_width=50, input_size=9, output_size=3, regression=True)
        basepath = '../refl/results'
        fname = os.path.join(basepath, 'ee75c74afd0e49dc9a93f08d2ff489c2/best_model.pth')
        self.model.load_state_dict(torch.load(fname, map_location=torch.device('cpu')))
        self.model.eval()
        self.nnDigits = Belle2.PyStoreArray(Belle2.TOPDigit.Class(), "NNTOPDigits")
        self.nnDigits.registerInDataStore()



    def event(self):
        simPhotons = Belle2.PyStoreArray("TOPSimPhotons")
        photons = []
        moduleIDs = []
        for simp in simPhotons:
            # define inputs
            ep = simp.getEmissionPoint()
            ed = simp.getEmissionDir()
            t = simp.getEmissionTime()
            e = simp.getEnergy() * 1.0E-9 # direct output from .getEnergy(), in GeV so around 2-3E-9
            x = ep.X() # in cm in local frame
            y = ep.Y() # in cm in local frame
            z = ep.Z() # in cm in local frame
            px = ed.X()*e # direct output from .getMomentum().X(), usually around 1.0E-9
            py = ed.Y()*e # direct output from .getMomentum().Y(), usually around 1.0E-9
            pz = ed.Z()*e # direct output from .getMomentum().Z(), usually around 1.0E-9
            rand = 0.5 # between 0 and 1
            photons.append([t, x, y, z, px, py, pz, e, rand])
            moduleIDs.append(simp.getModuleID())

        X = torch.Tensor(photons)
        Y = self.model(X).detach().numpy()
        for idx, p in enumerate(Y):
            nnDigit = self.nnDigits.appendNew()
            nnDigit.setPixelID(int(p[0]*64) + int(64*p[1]*8) + 1)
            nnDigit.setModuleID(moduleIDs[idx])
            nnDigit.setTime(p[2]*50-t)
        # print(Y)
        # print(Y[0]*64, Y[1]*8, Y[2]*50-t) # column 1 is normalized x pixel, column 2 is normalized y pixel
           # column 3 is normalized t bin (between 0 and 50 ns)


class DigitPrinter(Module):
    def event(self):
        topDigits = Belle2.PyStoreArray("TOPDigits")
        for t in topDigits:
            print(t.getPixelCol(), t.getPixelRow(), t.getTime())
        newDigits = Belle2.PyStoreArray("NNTOPDigits")
        for t in newDigits:
            print(t.getPixelCol(), t.getPixelRow(), t.getTime())


# Suppress messages and warnings during processing:
set_log_level(LogLevel.ERROR)

# channel mask
#if opts.local_db:
#    use_local_database("localDB/localDB.txt", "localDB", False)

# Create path
main = create_path()

# Set number of events to generate
eventinfosetter = register_module('EventInfoSetter')
# Number of events reset to one in evtNumList: [1]
eventinfosetter.param({'evtNumList': [10000], 'runList': [int(os.environ.get("LSB_JOBINDEX", "1"))]})
main.add_module(eventinfosetter)

# Histogram manager immediately after master module
histo = register_module('HistoManager')
histo.param('histoFileName', 'DQMhistograms.root')  # File to save histograms
main.add_module(histo)

# Gearbox: access to database (xml files)
gearbox = register_module('Gearbox')
main.add_module(gearbox)

# Geometry
geometry = register_module('Geometry')
main.add_module(geometry)
geometry.param('components', [
        'MagneticField',
        'BeamPipe',
        'PXD',
        'SVD',
        'CDC',
        'TOP',
])

# Particle gun: generate multiple tracks
particlegun = register_module('ParticleGun')
#particlegun.param('pdgCodes', [opts.particle])
particlegun.param('pdgCodes', [13, -13])
# TODO +/-
particlegun.param('nTracks', 1)
particlegun.param('varyNTracks', False)
particlegun.param('independentVertices', False)

particlegun.param('momentumGeneration', 'uniform')
#particlegun.param('momentumParams', [opts.momentum, opts.momentum])
particlegun.param('momentumParams', [1.0, 3.0])
# Added arguments for particle gun parameters (momentum, xyz, theta, phi) using opts
#particlegun.param('thetaGeneration', 'uniformCos')
particlegun.param('thetaGeneration', 'uniform')
#particlegun.param('thetaParams', [opts.theta, opts.theta])
particlegun.param('thetaParams', [32.6, 122])
particlegun.param('phiGeneration', 'uniform')
#particlegun.param('phiParams', [opts.phi, opts.phi])
particlegun.param('vertexGeneration', 'fixed')
particlegun.param('xVertexParams', [opts.xVertex])
particlegun.param('yVertexParams', [opts.yVertex])
particlegun.param('zVertexParams', [opts.zVertex])


main.add_module(particlegun)
# Simulation
simulation = register_module('FullSim')
main.add_module(simulation)

add_svd_simulation(main)
add_pxd_simulation(main)

# PXD digitization & clustering
add_pxd_reconstruction(main)

# SVD digitization & clustering
add_svd_reconstruction(main)

# CDC digitization
cdcDigitizer = register_module('CDCDigitizer')
main.add_module(cdcDigitizer)

# TOP digitization
topdigi = register_module('TOPDigitizer')
main.add_module(topdigi)

add_tracking_reconstruction(main)

# Track extrapolation
ext = register_module('Ext')
main.add_module(ext)

# TOP reconstruction
top_cm = register_module('TOPChannelMasker')
main.add_module(top_cm)

nnt = NN_TOPDigitMaker()
main.add_module(nnt)

dp = DigitPrinter()
main.add_module(dp)

topreco = register_module('TOPReconstructor')
topreco.TOPDigitCollectionName = "NNTOPDigits"
#topreco.logging.log_level = LogLevel.DEBUG  # remove or comment to suppress printout
#topreco.logging.debug_level = 2  # or set level to 0 to suppress printout
main.add_module(topreco)


# TOP DQM
#topdqm = register_module('TOPDQM')
#main.add_module(topdqm)

# Show progress of processing
progress = register_module('Progress')
main.add_module(progress)

# Process events
process(main)

# Print call statistics
#print(statistics)
