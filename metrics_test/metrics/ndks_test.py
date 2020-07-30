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

import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import os
from glob import glob
home = os.getenv("HOME")


class ndKS(object):
    def __init__(self, soft=False):
        if soft:
            self.max = smooth_max()
        else:
            self.max = torch.max
        
    def __call__(self, pred, true, alternative):
        os_pp = self.get_octants(pred, pred, alternative)
        os_pt = self.get_octants(true, pred, alternative)
        D1 = self.max((os_pp - os_pt).abs())
        os_tt = self.get_octants(true, true, alternative)
        os_tp = self.get_octants(pred, true, alternative)
        D2 = self.max((os_tt - os_tp).abs())
        return (D1 + D2)/2.
    
    @staticmethod
    def squarewave(f, x):
        return (((-1)**torch.floor(f)+1)/2 - x).long().abs()
    
    @staticmethod
    def get_octants(x, points, alternative):
        N = x.shape[0]
        dim = x.shape[1]
        # shape our input and test points into the right shape (N, 3, 1)
        x = x.unsqueeze(-1)
        #print(x)
        points = points.unsqueeze(-1)
        # repeat each input point in the dataset across the third dimension
        x = x.repeat((1, 1, N))
        #print(x)
        # repeate each input point in the dataset across the first dimension
        comp_x = points.repeat((1, 1, N))
        comp_x = comp_x.permute((2, 1, 0))
        #print(comp_x)
        # now compare the input points and comparison points to see how many
        # are bigger and smaller
        x = torch.ge(x, comp_x).long()
        #print(x)
        #print(x[:, 0, :])
        nx = (1 - torch.clone(x)).abs()
        freq = torch.tensor([[i/2**(j) for j in range(dim)] for i in range(2**dim)])
        stack_tens = torch.empty((2**dim, N))
        #print(nx)
        if alternative:
            o1 = torch.sum(torch.prod(x, dim = 1), dim=0).float() / N
            # return the stack of octants, should be (n, 8)
            return torch.stack([o1, o2, o3, o4, o5, o6, o7, o8], dim=1)
            for i in range(2**dim):
                stack_tens[i] = torch.sum(ndKS.squarewave(freq[i][0], x[:, 0, :]) &  ndKS.squarewave(freq[i][1], x[:, 1, :]) &  ndKS.squarewave(freq[i][2], x[:, 2, :]), dim=0).float() / N
            print(stack_tens.T)
            return(stack_tens)
        else:
            # now use the comparisoned points to construct each octant (& is logical and)
            o1 = torch.sum( x[:, 0, :] &  x[:, 1, :] &  x[:, 2, :], dim=0).float() / N
            o2 = torch.sum( x[:, 0, :] &  x[:, 1, :] & nx[:, 2, :], dim=0).float() / N
            o3 = torch.sum( x[:, 0, :] & nx[:, 1, :] &  x[:, 2, :], dim=0).float() / N
            o4 = torch.sum( x[:, 0, :] & nx[:, 1, :] & nx[:, 2, :], dim=0).float() / N
            o5 = torch.sum(nx[:, 0, :] &  x[:, 1, :] &  x[:, 2, :], dim=0).float() / N
            o6 = torch.sum(nx[:, 0, :] &  x[:, 1, :] & nx[:, 2, :], dim=0).float() / N
            o7 = torch.sum(nx[:, 0, :] & nx[:, 1, :] &  x[:, 2, :], dim=0).float() / N
            o8 = torch.sum(nx[:, 0, :] & nx[:, 1, :] & nx[:, 2, :], dim=0).float() / N
            # return the stack of octants, should be (n, 8)
            return torch.stack([o1, o2, o3, o4, o5, o6, o7, o8], dim=1)
    
    def permute(self, J=1_000):
        all_pts = torch.cat((self.pred, self.true), dim=0)
        T = self(self.pred, self.true)
        T_ = torch.empty((J,))
        total_shape = self.pred.shape[0] + self.true.shape[0]
        for j in range(J):
            idx = torch.randperm(total_shape)
            idx1, idx2 = torch.chunk(idx, 2)
            _pred = all_pts[idx1]
            _true = all_pts[idx2]
            T_[j] = self(_pred, _true)
        return torch.sum(T_ > T) / float(J), T, T_


pred = [[1, 7, 1], [18, 5, -1]]
true = [[1, 3, 4], [4, 5, 2]]
true = torch.from_numpy(np.asarray(true))
pred = torch.from_numpy(np.asarray(pred))
cls1 = ndKS()
alternative=True
# %time print(cls1(pred, true, alternative))
cls2 = ndKS()
alternative=False
# %time print(cls2(pred, true, alternative))

cls1 = ndKS()
alternative=True
# %time print(cls1(pred, true, alternative))

cls2 = ndKS()
alternative=True
# %time print(cls2(pred, true, alternative))

N = 1000
pred = torch.normal(mean=0.0, std=1.0, size=(N, 3))
true = torch.normal(mean=0.0, std=1.0, size=(N, 3))
cls = ndKS()
alternative = True
# %time print(cls(pred, true, alternative))

cls = ndKS()
alternative = False
# %time print(cls(pred, true, alternative))

# +
mus = np.linspace(-2.0, 2.0)
Ds = np.empty((mus.shape[0]))
Ds2 = np.empty((mus.shape[0]))
for i, mu in enumerate(mus):
    pred = torch.normal(mean=mu, std=1.0, size=(N, 3))
    Ds[i] = cls(pred, true, True)
    Ds2[i] = cls(pred, true, False)

plt.xlabel(r'Mean ($\mu$)')
plt.ylabel('Test Statistic ($D$)')
plt.plot(mus, Ds, 'r', label = 'My method')
plt.plot(mus, Ds2, 'b', label = 'Direct implementation')
plt.legend()
#plt.savefig(home + '/Documents/Masterarbeit-master/metrics_test/plots/metrics/ndKS/compare_orig.pdf')
plt.show()
# -

plt.xlabel(r'Mean ($\mu$)')
plt.ylabel('Test Statistic ($D$)')
plt.plot(mus, Ds, 'r', label = 'My method')
plt.plot(mus, Ds2, 'b', label = 'Direct implementation')
plt.legend()
#plt.savefig(home + '/Documents/Masterarbeit-master/metrics_test/plots/metrics/ndKS/compare_orig.pdf')
plt.show()
