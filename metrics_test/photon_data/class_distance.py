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
        
	def variables(self, evt1, evt2, photons):
		event_sample1 = np.random.choice(np.arange(0, len(evt1)), photons)
		event_sample2 = np.random.choice(np.arange(0, len(evt2)), photons)
		self.x1 = torch.from_numpy((evt1.iloc[event_sample1,:].detection_pixel_x.to_numpy() + self.xmax)/(-self.xmin+self.xmax))
		self.y1 = torch.from_numpy((evt1.iloc[event_sample1,:].detection_pixel_y.to_numpy() + abs(self.ymin))/(self.ymax-self.ymin))
		self.x2 = torch.from_numpy((evt2.iloc[event_sample2,:].detection_pixel_x.to_numpy() + self.xmax)/(-self.xmin+self.xmax))
		self.y2 = torch.from_numpy((evt2.iloc[event_sample2,:].detection_pixel_y.to_numpy() + abs(self.ymin))/(self.ymax-self.ymin))
		self.t1 = torch.from_numpy((evt1.iloc[event_sample1,:].detection_time.to_numpy())/(self.tmin+self.tmax))
		self.t2 = torch.from_numpy((evt2.iloc[event_sample2,:].detection_time.to_numpy())/(self.tmin+self.tmax))
        
	def compute_dist(self, evt1, evt2, p=True):
		if p:
			p1 = np.asarray((evt1.production_px.mean(axis=0), evt1.production_py.mean(axis=0), evt1.production_pz.mean(axis=0)))
			p2 = np.asarray((evt2.production_px.mean(axis=0), evt2.production_py.mean(axis=0), evt2.production_pz.mean(axis=0)))
			return np.sum((p1 - p2)**2)
		else:
			x1 = np.asarray((evt1.production_x.mean(axis=0), evt1.production_y.mean(axis=0), evt1.production_z.mean(axis=0)))
			x2 = np.asarray((evt2.production_x.mean(axis=0), evt2.production_y.mean(axis=0), evt2.production_z.mean(axis=0)))
			return np.sum((x1 - x2)**2)
    
	def __call__(self, evt1, evt2, photons):
		self.variables(evt1, evt2, photons)
		cls = ndKS()
		if self.dim == 3:
			self.dist = cls(torch.stack((self.x1, self.y1, self.t1), axis=-1),
                            torch.stack((self.x2, self.y2, self.t2), axis=-1)).item()
			return(self.dist)
		if self.dim == 2:
			self.dist = [cls(torch.stack((self.x1, self.y1), axis=-1), torch.stack((self.x2, self.y2), axis=-1)).item(), 
                         cls(torch.stack((self.x1, self.t1), axis=-1), torch.stack((self.x2, self.t2), axis=-1)).item(),
                        cls(torch.stack((self.t1, self.y1), axis=-1), torch.stack((self.t2, self.y2), axis=-1)).item()]
			return(self.dist)
