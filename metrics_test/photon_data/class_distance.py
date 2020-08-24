# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import torch
import sys
from ndks import ndKS


class get_dstc(object):
    def __init__(self, dim=3, xmin=-22.5, xmax=22.5,
                ymin = -4.5, ymax=1.0, tmin=0.0, tmax = 70.0):
        self.dim = dim
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.tmin = tmin
        self.tmax = tmax
        
    @staticmethod
    def variables(evt1, evt2, photons):
        event_sample1 = np.random.choice(np.arange(0, len(evt1)), photons)
        event_sample2 = np.random.choice(np.arange(0, len(evt2)), photons)
        x1 = torch.from_numpy((evt1.iloc[event_sample1,:].detection_pixel_x.to_numpy() + self.xmax)/(-self.xmin+self.xmax))
        y1 = torch.from_numpy((evt1.iloc[event_sample1,:].detection_pixel_y.to_numpy() + abs(self.ymin))/(self.ymax-self.ymin))
        x2 = torch.from_numpy((evt2.iloc[event_sample2,:].detection_pixel_x.to_numpy() + self.xmax)/(-self.xmin+self.xmax))
        y2 = torch.from_numpy((evt2.iloc[event_sample2,:].detection_pixel_y.to_numpy() + abs(self.ymin))/(self.ymax-self.ymin))
        t1 = torch.from_numpy((evt1.iloc[event_sample1,:].detection_time.to_numpy())/(self.tmin+self.tmax))
        t2 = torch.from_numpy((evt2.iloc[event_sample2,:].detection_time.to_numpy())/(self.tmin+self.tmax))
        return (x1, y1, t1, x2, y2, t2)

    def __call__(self, evt1, evt2, photons):
        x1, y1, t1, x2, y2, t2 = self.variables(evt1, evt2, photons)
        cls = ndKS()
        if dim == 3:
            self.dist = cls(torch.stack((x1, y1, t1), axis=-1),
                            torch.stack((x2, y2, t2), axis=-1))
        if dim == 2:
            self.dist = [cls(torch.stack((x1, y1), axis=-1), torch.stack((x2, y2), axis=-1)), 
                         cls(torch.stack((x1, t1), axis=-1), torch.stack((x2, t2), axis=-1)),
                        cls(torch.stack((t1, y1), axis=-1), torch.stack((t2, y2), axis=-1))]
        return self.dist
