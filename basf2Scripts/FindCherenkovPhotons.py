from basf2 import *
from ROOT import Belle2
from pandas import DataFrame
import pandas as pd


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
        store = pd.HDFStore(f'/ceph/ihaide/ogun/Gauss/{self.fname}_{self.jobid}.h5', complevel=9, complib='blosc:lz4')
        store["photons"] = dfphotons
        store.close()
