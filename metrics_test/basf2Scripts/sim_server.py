import basf2 as b2
import ROOT
from ROOT import Belle2
import os
import zmq
import pandas as pd
import sys
import logging
import uuid


def empty_func(self):
    return

class FindCherenkovPhotons(b2.Module):
    def __init__(self, server):
        super().__init__()
        self.__name__ = 'FCP'
        self.server = server

    def initialize(self):
        self.filename = 'something.h5'
        self.index = 0
        

    def event(self):
        self.photons = []
        photons = Belle2.PyStoreArray("TOPDetectedPhotons")
        if len(photons) == 0:
            mcp = Belle2.PyStoreArray("MCParticles")
            m = mcp[0]
            self.photons.append(
                                (
                                 self.index,
                                 m.getProductionTime(), #production_time,
                                 m.getVertex().X(), #production_x,
                                 m.getVertex().Y(), #production_y,
                                 m.getVertex().Z(), #production_z,
                                 m.getMomentum().X(), #production_px,
                                 m.getMomentum().Y(), #production_py,
                                 m.getMomentum().Z(), #production_pz,
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
            #print("emission x y z", production_x, production_y, production_z)
            momentum = p.getEmissionDir()
            production_px = momentum.X()
            production_py = momentum.Y()
            production_pz = momentum.Z()
            production_e = p.getEnergy()
            #print("emission px py pz e", production_px, production_py, production_pz, production_e)
            detection_point = p.getDetectionPoint()
            detection_x = detection_point.X()
            detection_y = detection_point.Y()
            detection_z = detection_point.Z()
            #print("detection x y z", detection_x, detection_y, detection_z)
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
        dfphotons = pd.DataFrame(data=self.photons, columns=photonColNames)
        logging.debug("Simulated, going to send dataframe")
        #print(f"evt_idx: {dfphotons.tail(1)['evt_idx']}", flush=True)
        #print(f"Production_x: {dfphotons.tail(1)['production_x']}", flush=True)
        self.server.send_df(dfphotons)

    def terminate(self):
        logging.debug("terminating")

class SimulationServiceInput(b2.Module):
    def __init__(self, path):
        super().__init__()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        # self.socket.bind("tcp://*:23553")
        self.socket.bind("tcp://*:237711")
        self.__name__ = 'SSI'
        self.path = path
        
    def initialize(self):
        logging.debug("Running SSI initialize")
        self.opticalgun = b2.register_module('OpticalGun')
        #self.path.add_module(self.opticalgun)
        
    def event(self):
        logging.debug("Running SSI Event")
        # get input values from zmq, blocking
        message = self.socket.recv_json()
        if ('cmd' in message) and (message['cmd'] == 'terminate'):
            for module in self.path.modules():
                if str(module) == 'EventInfoSetter':
                    logging.debug("turning off the EventInfoSetter")
                    module.param('evtNumList', [1])
                    #module.param('resetEvery', False)
        else:
            for key, val in message.items():
                logging.debug(f"Setting {key} to {val}.")
                logging.debug(type(key), type(val))
                self.opticalgun.param(key, val)
        code = 0
        self.socket.send_string(f"RECV{code}")
        self.opticalgun.event()

    def send_df(self, df):
        # create an in memory store
        uf = str(uuid.uuid4())
        store = pd.HDFStore(f'{uf}.h5', driver='H5FD_CORE', driver_core_backing_store=0)
        #store = pd.HDFStore(f'{self.fname}_{self.jobid}.h5', complevel=9, complib='blosc:lz4')
        store["photons"] = df
        #print(f"in sending df production_x: {df.tail(1)['production_x']}", flush=True)
        binmsg = store._handle.get_file_image()
        store.close()
        message = self.socket.recv_string()
        logging.debug(f"received {message}")
        if message == 'ready':
            self.socket.send(binmsg)
        


p = b2.Path()
# now add an input module to wait for input, in this case stdin ... it will create events as necessary
eventinfosetter = b2.register_module('EventInfoSetter')
eventinfosetter.param('evtNumList', [2])
eventinfosetter.param('resetEvery', True)
p.add_module(eventinfosetter)

gearbox = b2.register_module('Gearbox')
p.add_module(gearbox)

geometry = b2.register_module('Geometry')
p.add_module(geometry)

# add all the simulation and necessary reconstruction in here
simserver = SimulationServiceInput(p)
p.add_module(simserver)

# Simulation
simulation = b2.register_module('FullSim')
simulation.param('PhotonFraction', 1.0)
p.add_module(simulation)

# TOP digitization
topdigi = b2.register_module('TOPDigitizer')
topdigi.param('useWaveforms', False)
topdigi.param('simulateTTS', False)
topdigi.param('electronicJitter', 0.0)
topdigi.param('timeZeroJitter', 0.0)
p.add_module(topdigi)

# Output
fcp = FindCherenkovPhotons(simserver)
fcp.jobid = 1
p.add_module(fcp)

# Show progress of processing
progress = b2.register_module('Progress')
p.add_module(progress)

# and then have an output module that sends whatever you are interested in out (in this case stdout)
b2.process(p)
