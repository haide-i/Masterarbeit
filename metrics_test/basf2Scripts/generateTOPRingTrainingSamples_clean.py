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
import sys
from datetime import datetime

jobID = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
if not jobID:
    jobID = int(os.environ.get("LSB_JOBINDEX", "0"))


# command line options
ap = ArgumentParser('')
ap.add_argument('--output', '-o', default=f'output_{jobID}.h5', help='output filename')
ap.add_argument('--particle',       type=int,   default=13,   help='pdg code of the particles to generate')
ap.add_argument('--momentum_lower', type=float, default=2.,   help='lower bound for momentum')
ap.add_argument('--momentum_upper', type=float, default=2.,   help='upper bound for momentum')
ap.add_argument('--phi_lower',      type=float, default=0.,   help='lower bound for phi')
ap.add_argument('--phi_upper',      type=float, default=360., help='upper bound for phi')
ap.add_argument('--theta_lower',    type=float, default=32.6, help='lower bound for theta')
ap.add_argument('--theta_upper',    type=float, default=122., help='upper bound for theta')
ap.add_argument('--xVertex',        type=float, default=0.,   help='x-coordinate of initial position')
ap.add_argument('--yVertex',        type=float, default=0.,   help='y-coordinate of initial position')
ap.add_argument('--zVertex',        type=float, default=0.,   help='z-coordinate of initial position')
ap.add_argument('--photonFraction', type=float, default=1.,   help='photon fraction')
ap.add_argument('--randomSeed',     type=int,   default=0,    help='random seed')

opts = ap.parse_args()
print("random_seed: {:d}".format(int(opts.randomSeed)))

class FindCherenkovPhotons(Module):
    def initialize(self):
        self.jobid = jobID
        self.index = 0 # self.jobid * 100000
        self.indices = []
        self.particles = []
        self.exts = []
        self.photons = []
        self.KaonPDFPeaks = []
        self.muonPDFPeaks = []
        self.electronPDFPeaks = []
        self.pionPDFPeaks = []
        self.debugPhotons = []
        self.xAxis = ROOT.TVector3(1, 0, 0)
        self.yAxis = ROOT.TVector3(0, 1, 0)
        self.zAxis = ROOT.TVector3(0, 0, 1)

    def event(self):
        digits = Belle2.PyStoreArray("TOPDigits")
        numdigits = len(digits)

        photons = Belle2.PyStoreArray("TOPSimPhotons")
        debugPhotons = Belle2.PyStoreArray("TOPDebugPhotons")
        mcps = Belle2.PyStoreArray("MCParticles")
        bHits = Belle2.PyStoreArray("TOPBarHits")
        eHits = Belle2.PyStoreArray("ExtHits")
        tophits = [x for x in eHits if x.getDetectorID() == Belle2.Const.TOP] # or 0x4
        topenters = [x for x in tophits if x.getStatus() == 0] # or 0
        topexits = [x for x in tophits if x.getStatus() == 1] # or 1

#        if not topenters:
#            return
#        if not topexits:
#            return

#        eHit_enter = sorted(topenters, key=lambda x: x.getTOF())[0]
#        eHit_exit  = sorted(topexits, key=lambda x: x.getTOF())[0]

#        if eHit_enter == None:
#            print('no enter ext hit'); return
#        if eHit_exit == None:
#            print('no exit ext hit'); return

        # handle hadronic interactions by only considering the particle with
        # maximum energy (which should always be the initial particle)
        mcp = sorted(mcps, key=lambda x: x.getEnergy(), reverse=True)[0]

        # selected the mcp's related TOPLikelihood
        track = mcp.getRelated('Tracks')
        if not track:
            print('no related track'); return
        logl = track.getRelated('TOPLikelihoods')
        if not logl:
            print('no related loglihood'); return

        # handle backscatter by only considering barhits from the initial particle
        for b in bHits:
            if b.getProductionPoint().Mag() < 0.01:
                barhit = b
                break
        else:
            print('no bar hit from generated particle'); return

        hits = Belle2.PyStoreArray("TOPSimHits")
        for h in hits:
            m = h.getRelated("MCParticles")
            if not m:   continue
#            else:       print(m.getPDG(), '\t', m.getVertex().Mag())

        # reldigs = mcp.getRelated('TOPDigits')
        # if not reldigs:
        #     print('mcp has no related digits')
        # else:
        #     for d in digits:
        #         try:
        #             for reld in reldigs:
        #                 if d == reld: print('n')
        #         except:
        #             if d == reldigs: print('1')

        if not 7 < numdigits:
            print('less than eight digits: %d' % len(digits))

        # for diagnostics
        total_energy = sum([x.getEnergy() for x in mcps])
        phot_count = sum(1 for x in mcps if x.getPDG() == 22)

        localP = ROOT.TVector3()
        localP.SetMagThetaPhi(1, barhit.getTheta(), barhit.getPhi())
        for phot in debugPhotons:
            p = phot.getEmissionPoint()
            d = phot.getEmissionDir()
            cherenkovAngle = localP.Angle(d)
            self.debugPhotons.append(
                (
                    self.index,
                    phot.getDetectionTime(),
                    phot.getEmissionTime(),
                    p.X(), p.Y(), p.Z(),
                    d.X(), d.Y(), d.Z(),
                    cherenkovAngle
                )
            )
        moduleID = 0
        for d in digits:
            if moduleID == 0:
                moduleID = d.getModuleID()
            simhit = d.getRelated("TOPSimHits")
            if not simhit:
                continue
            photon = simhit.getRelated("TOPSimPhotons")
            if not photon:
                continue

            origin = 0

            # we define a new coordinate system:
            # our phi is in the x-z plane, which tells us if the photon goes forward or backward
            # our theta is the angle with the y-axis, which tells us the number of reflections on the top and bottom surfaces
            point = photon.getEmissionPoint()
            dir = photon.getEmissionDir()
            cherenkovAngle = localP.Angle(dir)
            photonInXZPlane = ROOT.TVector3(dir)
            localTheta = photonInXZPlane.Angle(self.yAxis)
            photonInXZPlane.SetY(0)
            localPhi = np.sign(photonInXZPlane.X()) * photonInXZPlane.Angle(self.zAxis)
            #print(point.Mag())
            self.photons.append(
                (
                    self.index,
                    d.getPixelID(),
                    photon.getDetectionTime(),
                    origin,
                    point.X(), point.Y(), point.Z(),
                    dir.X(), dir.Y(), dir.Z(),
                    (photon.getDetectionTime() - photon.getEmissionTime()),
                    cherenkovAngle,
                    localTheta, localPhi,
                )
            )
        print("photon length", len(self.photons))
        if mcp is None:
            print('mcp is none'); return
        self.indices.append(self.index)
        self.particles.append(
            (
                self.index,
                moduleID * mcp.getCharge(),
                mcp.getMomentum().X(), mcp.getMomentum().Y(), mcp.getMomentum().Z(),
                barhit.getLocalPosition().X(), barhit.getLocalPosition().Y(), barhit.getLocalPosition().Z(),
                total_energy, phot_count,
                logl.getNphot(), logl.getEstBkg(), logl.getFlag(),
                logl.getLogL_e(), logl.getLogL_mu(), logl.getLogL_pi(), logl.getLogL_K(), logl.getLogL_p(),
                logl.getNphot_e(), logl.getNphot_mu(), logl.getNphot_pi(), logl.getNphot_K(), logl.getNphot_p(),
                numdigits,
                #eHit_exit.getTOF() - eHit_enter.getTOF(),
                #eHit_enter.getPosition().X(), eHit_enter.getPosition().Y(), eHit_enter.getPosition().Z(),
                #eHit_exit.getPosition().X(), eHit_exit.getPosition().Y(), eHit_exit.getPosition().Z()
            )
        )
        self.index += 1


    def terminate(self):
        photonColNames = ("mcpidx",
                         "pixelID",
                         "time",
                         "origin",
                         "x", "y", "z",
                         "px", "py", "pz",
                         "flight_time",
                         "thetaC",
                         "localTheta", "localPhi"
                        )
        dfphotons = DataFrame(data=self.photons, columns=photonColNames)
        partColNames = ("mcpidx", "moduleID",
                        "px", "py", "pz",
                        "lx", "ly", "lz",
                        "E_tot", "phot_ct",
                        "nPhot", "estBG", "flag",
                        "loglE", "loglMu", "loglPi", "loglK", "loglP",
                        "nPhotE", "nPhotMu", "nPhotPi", "nPhotK", "nPhotP",
                        "nDigits", 
                        #"TOPtime",
                        #"enterx", "entery", "enterz",
                        #"exitx", "exity", "exitz"
                       )
        dfparticles = DataFrame(data=self.particles, columns=partColNames, index=self.indices)
        debugPhotonColNames = ("mcpidx",
                         "detectionTime",
                         "emissionTime",
                         "x", "y", "z",
                         "px", "py", "pz",
                         "thetaC"
                        )
        dfDebugPhotons = DataFrame(data=self.debugPhotons, columns=debugPhotonColNames)
        #store = pd.HDFStore(opts.output, complevel=9, complib='blosc:lz4')
        #store["photons"] = dfphotons
        #store["mcp"] = dfparticles
        #store['debugPhotons'] = dfDebugPhotons
        #store.close()
        print("n_photons: {:d}".format(int(len(self.photons))))
        print("n_debug_photons: {:d}".format(int(len(self.debugPhotons))))

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
eventinfosetter.param({'evtNumList': [10000], 'runList': [0]})
main.add_module(eventinfosetter)

# Gearbox: access to database (xml files)
gearbox = register_module('Gearbox')
main.add_module(gearbox)

# Geometry (only TOP and B-field)
geometry = register_module('Geometry')
geometry.param('useDB', False)
geometry.param('components', ['MagneticField', 'TOP'])
main.add_module(geometry)

# Histogram manager immediately after master module
histo = register_module('HistoManager')
histo.param('histoFileName', 'DQMhistograms_%d.root' % jobID)  # File to save histograms
main.add_module(histo)


# Particle gun: generate multiple tracks
particlegun = register_module('ParticleGun')
particlegun.param('pdgCodes', [opts.particle])
particlegun.param('nTracks', 1)
particlegun.param('varyNTracks', False)
particlegun.param('independentVertices', False)
particlegun.param('momentumGeneration', 'uniform')
particlegun.param('momentumParams', [opts.momentum_lower, opts.momentum_upper])
particlegun.param('thetaGeneration', 'uniform') # 'uniformCos'
particlegun.param('thetaParams', [opts.theta_lower, opts.theta_upper])
particlegun.param('phiGeneration', 'uniform')
particlegun.param('phiParams', [opts.phi_lower, opts.phi_upper])
particlegun.param('vertexGeneration', 'fixed')
particlegun.param('xVertexParams', [opts.xVertex])
particlegun.param('yVertexParams', [opts.yVertex])
particlegun.param('zVertexParams', [opts.zVertex])



main.add_module(particlegun)
# Simulation
simulation = register_module('FullSim')
simulation.param('PhotonFraction', opts.photonFraction)
main.add_module(simulation)

# TOP digitization: all time jitters turned OFF
topdigi = register_module('TOPDigitizer')
topdigi.param('useWaveforms', False)
topdigi.param('simulateTTS', False)
topdigi.param('electronicJitter', 0.0)
topdigi.param('timeZeroJitter', 0.0)
main.add_module(topdigi)

# Dedicated track maker using MC information only
trackmaker = register_module('TOPMCTrackMaker')
main.add_module(trackmaker)

# TOP PDF: time jitters are excluded
toppdf = register_module('TOPPDFChecker')
main.add_module(toppdf)

# Show progress of processing
progress = register_module('Progress')
main.add_module(progress)

topreco = register_module('TOPReconstructor')
#topreco.logging.log_level = LogLevel.DEBUG  # remove or comment to suppress printout
#topreco.logging.debug_level = 2  # or set level to 0 to suppress printout
main.add_module(topreco)

fcp = FindCherenkovPhotons()
fcp.fname = opts.output
main.add_module(fcp)

# Show progress of processing
progress = register_module('Progress')
main.add_module(progress)

# Process events
start = datetime.now()
process(main)
end = datetime.now()
time_to_simulate = (end - start).total_seconds()
print("time_to_simulate: {:d}".format(int(time_to_simulate)))

# Print call statistics
#print(statistics)
