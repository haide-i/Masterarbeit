import numpy as np

x = np.random.rand()*45 - 22.5
y = np.random.rand()*2 - 1
z = 0
psi = np.random.rand()*360
theta = np.random.rand()*360
phi = np.random.rand()*360

data = np.array([x, y, z, psi, theta, phi])
#np.savetxt('startdata.txt', data)
data
