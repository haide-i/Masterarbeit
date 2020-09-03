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

# example of using OpticalGun to simulate laser light source
# laser source at the mirror (inside bar) emitting toward PMT's
# ---------------------------------------------------------------

class FindCherenkovPhotons(Module):
    def initialize(self):
        self.photons = []
        self.len = 0

    def event(self):
        photons = Belle2.PyStoreArray("TOPSimPhotons")
        mcp = Belle2.PyStoreArray("MCParticles")
        #print(dir(mcp))
        #print(dir(mcp[0]))
        #print(mcp[0].getSeenInDetector())
        self.len += len(mcp)

    def terminate(self):
        print(f"Total photons {self.len}.")

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
main.add_module(geometry)

# Optical gun
opticalgun = register_module('OpticalGun')
opticalgun.param('angularDistribution', 'cone')
opticalgun.param('track', '-22.4, 0.0, 0.0, 0.0; 22.4, 0.0, 0.0, 0.01')
#opticalgun.param('minAlpha', 30.0)
#opticalgun.param('maxAlpha', 45.0)
#opticalgun.param('startTime', 0.0)
#opticalgun.param('pulseWidth', 0.0)
opticalgun.param('numPhotons', 100_000)
#opticalgun.param('diameter', 0.0)
#opticalgun.param('slotID', 1)  # if nonzero, local (bar) frame, otherwise Belle II
#opticalgun.param('positionDistribution', 'fixed')
#opticalgun.param('directionDistribution', 'cone')
main.add_module(opticalgun)

# Simulation
simulation = register_module('FullSim')
simulation.param('PhotonFraction', 1.0)
main.add_module(simulation)

# TOP digitization
topdigi = register_module('TOPDigitizer')
topdigi.param('useWaveforms', False)
topdigi.param('simulateTTS', False)
topdigi.param('electronicJitter', 0.0)
topdigi.param('timeZeroJitter', 0.0)
main.add_module(topdigi)

fcp = FindCherenkovPhotons()
main.add_module(fcp)

# Show progress of processing
progress = register_module('Progress')
main.add_module(progress)

# Process events
process(main)

# Print call statistics
print(statistics)
