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
def rotX(a):
    rotmat = [[1, 0, 0],
                      [0, np.cos(a), -np.sin(a)],
                      [0, np.sin(a), np.cos(a)]]
    return rotmat

def rotZ(a):
    rotmat = [[np.cos(a), -np.sin(a), 0],
                      [np.sin(a), np.cos(a), 0],
                      [0, 0, 1]]

    return rotmat

def get_momenta(psi, theta, phi, v=[0, 0, 1.]):
    psi = psi * math.pi/180.
    theta = theta * math.pi/180.
    phi = phi * math.pi/180.
    return np.dot(rotZ(psi), np.dot(rotX(theta), np.dot(rotZ(phi), v)))

def get_angle(momentum): #get the euler angles psi>270 and theta>180, if starting vector is (0, 0, 1)
    theta = np.arccos(-momentum[2]) + math.pi
    psi = np.arcsin(momentum[0]/np.sin(theta)) + 2*math.pi
    psi2 = np.arccos(momentum[1]/np.sin(theta)) + math.pi
    psi = 180.*psi/math.pi
    theta = 180.*theta/math.pi
    return theta, psi
