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
from FindCherenkovPhotons import FindCherenkovPhotons

def randOpticalGun(output_filename, jobID, sigma = 0.0, num_photons = 100, var = 'x', x=5.0, y=0.0, z=0.0, phi=0.0, theta=180.0, psi=0.0):
    
    wavelength = 405.0
    
        # example of using OpticalGun to simulate laser light source
    # laser source at the mirror (inside bar) emitting toward PMT's
    # ---------------------------------------------------------------

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
    # histo = register_module('HistoManager')
    # histo.param('histoFileName', f'DQMhistograms{jobID}.root')  # File to save histograms
    # main.add_module(histo)

    # Optical gun
    for i in range(5):
        mu = 0.2 * i//20
        print(mu)
        if var == 'x':
            x = np.random.normal(x + mu, sigma)
        elif var == 'y':
            y = np.random.normal(y + mu, sigma)
        elif var == 'z':
            z = np.random.normal(z + mu, sigma)
        elif var == 'phi':
            phi = np.random.normal(phi + mu, sigma)
        elif var == 'theta':
            theta = np.random.normal(theta + mu, sigma)
        elif var == 'psi':
            psi = np.random.normal(psi + mu, sigma)
        opticalgun = register_module('OpticalGun')
        opticalgun.param('angularDistribution', 'uniform')
        opticalgun.param('minAlpha', 0.0)
        opticalgun.param('maxAlpha', 0.00001)
        opticalgun.param('startTime', 0.0)
        opticalgun.param('pulseWidth', 0.0)
        opticalgun.param('numPhotons', int(num_photons))
        opticalgun.param('diameter', 0.0)
        opticalgun.param('slotID', 5)  # if nonzero, local (bar) frame, otherwise Belle II
        opticalgun.param('x', float(x))
        opticalgun.param('y', float(y))
        opticalgun.param('z', float(z))
        opticalgun.param('phi', float(phi))
        opticalgun.param('theta', float(theta))
        opticalgun.param('psi', float(psi))
        opticalgun.param('wavelength', float(wavelength))
        main.add_module(opticalgun)
    

    print(x)


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

    #df = pd.read_hdf('./Gaussx10/{}_{}.h5'.format(output_filename, jobID))
    #print(df)
