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
import math


# +
class get_mom(object):
    
    def __init__(self):
        pass
    
@static_method
    def rotX(a):
        rotmat = [[1, 0, 0],
                          [0, np.cos(a), -np.sin(a)],
                          [0, np.sin(a), np.cos(a)]]
        return rotmat
@static_method
    def rotZ(a):
        rotmat = [[np.cos(a), -np.sin(a), 0],
                          [np.sin(a), np.cos(a), 0],
                          [0, 0, 1]]

        return rotmat

    def __call__(self, phi, theta, psi, v):
        psi = psi * math.pi/180.
        theta = theta * math.pi/180.
        phi = phi * math.pi/180.
        return np.dot(rotZ(psi), np.dot(rotX(theta), np.dot(rotZ(phi), v)))
