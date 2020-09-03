#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
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
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import sys
import os.path
# import model definition
sys.path.append('../..')
import refl
from refl.data import dt
from refl.model import mt
from refl.loss import lt
import refl.util as util
from refl.util import bunyan
import refl.vis as vis
import tqdm

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


class NN_Trainer(Module):
    def initialize(self):
        self.n_epochs=2
        self.batch_size=32
        self.n=int(1E5)
        self.n_repeat=1
        self.regression=True
        self.latent_depth=200
        self.latent_width=200
        self.lr=1.0E-3
        self.save=True
        self.histogram='softhist'
        if util.onrc():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.device = 'cpu'

        # instantiate a logger - this logger saves to the text file and also
        # writes to mlflow if it can connect
        self.log = bunyan(mlflow=False)
        # Proess these arguments
        # Aggregate the current hyper parameters
        self.hyper = dict(n=self.n, batch_size=self.batch_size, latent_width=self.latent_width, latent_depth=self.latent_depth,
                        regression=self.regression, n_repeat=self.n_repeat, lr=self.lr, n_epochs=self.n_epochs, histogram=self.histogram)
        # make our training dataset and validation dataset
        self.dset = refl.data.dt.PhotonDataset(n_photons=self.n, regression=self.regression,
                                            n_repeat=self.n_repeat, h5_path="../data/train_rand_photons.h5")
        N_train = float(len(self.dset))
        self.valdset = refl.data.dt.PhotonDataset(n_photons=int(0.2*self.n), validation=True,
                                                regression=self.regression, h5_path="../data/val_rand_photons.h5")
        self.N_val = float(len(self.valdset))
        self.conddset = refl.data.dt.PhotonDataset(conditional=True,
                                                n_photons=int(0.5*self.n),
                                                regression=True, h5_path="../data/cond_photons.h5")
        self.N_cond = float(len(self.conddset))
        plt.figure(figsize=(5, 15))
        plt.subplot(311)
        plt.hist(self.dset._data['detection_pixel_x'], bins=64)
        plt.subplot(312)
        plt.hist(self.dset._data['detection_pixel_y'], bins=8)
        plt.subplot(313)
        plt.hist(self.dset._data['detection_time'], bins=500)
        plt.savefig("histograms1.pdf")
        plt.subplot(311)
        plt.hist(self.dset.x, bins=64)
        plt.subplot(312)
        plt.hist(self.dset.y, bins=8)
        plt.subplot(313)
        plt.hist(self.dset.t, bins=500)
        plt.savefig("histograms2.pdf")
        # build a model
        self.model = mt.MLP(input_size=len(self.dset.columns), output_size=3,
                                latent_depth=self.latent_depth,
                                latent_width=self.latent_width,
                                regression=self.regression).to(self.device)
        # build an optimizer
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # instantiate our bins in x, y, z and make our softhist loss functions
        if self.histogram == 'softhist':
            hist_class = lt.SoftHistogram
        elif self.histogram == 'softkde':
            hist_class = lt.SoftKDE
        elif self.histogram == 'softtri':
            hist_class = lt.SoftTri
        binsx = torch.from_numpy((self.dset.binsx[:-1] + self.dset.binsx[1:])/2.0).unsqueeze(0).type(self.dset.precision).to(self.device)
        self.shx = hist_class(binsx)
        binsy = torch.from_numpy((self.dset.binsy[:-1] + self.dset.binsy[1:])/2.0).unsqueeze(0).type(self.dset.precision).to(self.device)
        self.shy = hist_class(binsy)
        binst = torch.from_numpy((self.dset.binst[:-1] + self.dset.binst[1:])/2.0).unsqueeze(0).type(self.dset.precision).to(self.device)
        self.sht = hist_class(binst)
        # use squared earth mover's distance for the loss function
        loss_fcn = lt.semd
        # find the number of batches
        self.n_batches = len(self.dset) // self.batch_size
        self.n_valbatches = len(self.valdset) // self.batch_size
        # five updates per epoch
        self.update_frequency = self.n_batches // 5
        # instantiate mlflow
        # mlflow.set_tracking_uri("http://b2ana.pnl.gov:5000")
        # mlflow.set_experiment("TOPGAN")
        # find the weights of x, y, and z - these should be equal now that
        # the input is normalized
        self.wx = 1.0 / np.power(np.max(self.dset.binsx) - np.min(self.dset.binsx), 2.0)
        self.wy = 1.0 / np.power(np.max(self.dset.binsy) - np.min(self.dset.binsy), 2.0)
        self.wt = 1.0 / np.power(np.max(self.dset.binst) - np.min(self.dset.binst), 2.0)
        # add the weights to our hyper parameters
        self.hyper['wx'] = self.wx
        self.hyper['wy'] = self.wy
        self.hyper['wt'] = self.wt
        # instantiate an index so that we can shuffle
        self.idx = np.arange(len(self.dset))
        self.validx = np.arange(len(self.valdset))
        self.val_semd = np.inf


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

        # log the hyper parameters
        self.log.log_param_dict(self.hyper)
        # start the actual training loop
        # print("Starting the training loop")
        experiment_id = 1
        run_id = 1
        result_directory = f'{experiment_id}/{run_id}/artifacts/'
        os.makedirs(result_directory, exist_ok=True)

        for epoch in range(self.n_epochs):
            # shuffle the training and validation set
            np.random.shuffle(self.idx)
            np.random.shuffle(self.validx)
            for i_batch in range(self.n_batches - 1):
                self.model.train()
                # actually shuffle
                idxs = self.idx[i_batch * self.batch_size:(i_batch + 1) * self.batch_size]
                # get the training batch and labels
                batch, x_labels, y_labels, t_labels = self.dset[idxs]
                # zero the gradients
                self.opt.zero_grad()
                # pass everything to GPU
                batch = batch.to(self.device)
                x_labels = x_labels.to(self.device)
                y_labels = y_labels.to(self.device)
                t_labels = t_labels.to(self.device)
                # run the model on this batch
                scores = self.model(batch)
                # get the loss using softhist and semd
                loss, loss_x, loss_y, loss_t \
                    = lt.get_loss(scores, x_labels, y_labels, t_labels, self.shx, self.shy, self.sht)
                # make our gradient and optimizer steps
                loss.backward()
                self.opt.step()
                # calculate step for logging
                step = epoch * self.n_batches + i_batch
                # log the losses
                self.log.log_metric('semd', loss.detach().cpu().item(), step=step)
                self.log.log_metric('semd_x', self.wx * loss_x.detach().cpu().item(), step=step)
                self.log.log_metric('semd_y', self.wy * loss_y.detach().cpu().item(), step=step)
                self.log.log_metric('semd_t', self.wt * loss_t.detach().cpu().item(), step=step)
                self.log.log_metric('epoch', epoch, step=step)
                self.log.log_metric('batch', i_batch, step=step)
                # now if it's on the update frequency, we want to evaluate on the validation
                # set and visualize
                if (i_batch % self.update_frequency == 0):
                    self.model.eval()
                    #torch.save(model.state_dict(),
                    #           '/qfs/projects/belle2gpu/users/hage581/refl/checkpoints/checkpoint.pth.tar')
                    #print("{:03d} | {:04d} | {:10.2f}".format(epoch, i_batch, loss.detach().cpu().item()))
                    # aggregate all the validation scores and true values
                    val_output = mt.get_output_on_dset(self.valdset, self.model,
                                                       self.batch_size, self.device)
                    val_output['name'] = 'val'
                    val_output['color'] = '#F4AA00' # garnet
                    val_output['colortrue'] = '#0081AB'; val_output['colorpred'] = '#A63F1E' #topaz, bronze
                    valoutput = val_output['output']
                    valpredx = val_output['predx']
                    valtruex = val_output['truex']
                    valpredy = val_output['predy']
                    valtruey = val_output['truey']
                    valpredt = val_output['predt']
                    valtruet = val_output['truet']
                    train_output = mt.get_output_on_dset(self.dset, self.model,
                                                         self.batch_size, self.device)
                    train_output['name'] = 'train'
                    train_output['color'] = '#0081AB'
                    train_output['colortrue'] = '#870150'; train_output['colorpred'] = '#719500'#garnet, emslgreen
                    trainpredx = train_output['predx']
                    trainpredx = train_output['truex']
                    trainpredy = train_output['predy']
                    traintruey = train_output['truey']
                    trainpredt = train_output['predt']
                    traintruet = train_output['truet']
                    cond_output = mt.get_output_on_dset(self.conddset, self.model,
                                                        self.batch_size, self.device)
                    cond_output['name'] = 'cond'
                    cond_output['color'] = '#870150'
                    cond_output['colortrue'] = '#502D7F'; cond_output['colorpred'] = '#D77600'#amethyst, copper
                    condpredx = cond_output['predx']
                    condtruex = cond_output['truex']
                    condpredy = cond_output['predy']
                    condtruey = cond_output['truey']
                    condpredt = cond_output['predt']
                    condtruet = cond_output['truet']
                    ## calculate semd on valset and if lowest, save the model
                    val_loss, vlx, vly, vlt \
                        = lt.get_loss(torch.from_numpy(valoutput).float().to(self.device),
                                      torch.from_numpy(valtruex).float().to(self.device),
                                      torch.from_numpy(valtruey).float().to(self.device),
                                      torch.from_numpy(valtruet).float().to(self.device), self.shx, self.shy, self.sht)
                    if val_loss < self.val_semd:
                        self.val_semd = val_loss
                        somefilename = "best_model.pth".format(epoch, i_batch)
                        filename = os.path.join(result_directory, somefilename)
                        torch.save(self.model.state_dict(),  filename)
                    ## plot histograms versus real histograms
                    bins = dict(binsx=self.dset.binsx, binsy=self.dset.binsy, binst=self.dset.binst)
                    filename = f"dists_{epoch:03d}_{i_batch:04d}"
                    filename = os.path.join(result_directory, filename)
                    vis.plot_pdfs(val_output, train_output,
                                  bins=bins, filename=filename, save=self.save)
                    filename = f'cond_dists_{epoch:03d}_{i_batch:04d}'
                    vis.plot_pdfs(cond_output, bins=bins,
                                  filename=filename, save=self.save)
                    ## plot the difference histograms
                    filename = f"delta_dists_{epoch:03d}_{i_batch:04d}"
                    filename = os.path.join(result_directory, filename)
                    vis.plot_delta_pdfs(val_output, bins=bins,
                                        save=self.save, filename=filename)
                    # plot the cdfs w/ the
                    filename = f"rand_cdfs_{epoch:03d}_{i_batch:04d}"
                    filename = os.path.join(result_directory, filename)
                    val_ks = vis.plot_cdfs(val_output, N=self.N_val, bins=bins,
                                           save=self.save, filename=filename)
                    self.log.log_metric_dict(val_ks, prefix='val_')
                    filename = f"cond_cdfs_{epoch:03d}_{i_batch:04d}"
                    filename = os.path.join(result_directory, filename)
                    cond_ks = vis.plot_cdfs(cond_output, N=self.N_cond, bins=bins,
                                            save=self.save, filename=filename)
                    self.log.log_metric_dict(cond_ks, prefix='cond_')
                    # plot the twod illustration
                    filename = f"cond_2d_{epoch:03d}_{i_batch:04d}"
                    filename = os.path.join(result_directory, filename)
                    vis.plot_twod(train_output, bins=bins, save=self.save, filename=filename)


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

nnt = NN_Trainer()
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





# if __name__ == "__main__":
#     # Read in the necessary training arguments
#     fc = argparse.ArgumentDefaultsHelpFormatter
#     parser = argparse.ArgumentParser(description='Train photon predictor.',
#                                      formatter_class=fc)
#     parser.add_argument('--n-epochs', type=int, default=100)
#     parser.add_argument('--batch-size', type=int, default=32)
#     parser.add_argument('--n', type=int, default=int(1E5))
#     parser.add_argument('--n-repeat', type=int, default=1)
#     parser.add_argument('--regression', type=bool, default=True)
#     parser.add_argument('--latent-depth', type=int, default=200)
#     parser.add_argument('--latent-width', type=int, default=200)
#     parser.add_argument('--lr', type=float, default=1.0E-3)
#     parser.add_argument('--histogram', type=str, default='softhist')
#     args = parser.parse_args()
#     train(n_epochs=args.n_epochs, batch_size=args.batch_size,
#           n=args.n, n_repeat=args.n_repeat, regression=args.regression,
#           latent_depth=args.latent_depth, latent_width=args.latent_width,
#           lr=args.lr, histogram=args.histogram)
