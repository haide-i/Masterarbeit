import numpy as np

x = np.random.rand()*45 - 22.5
y = 0
z = np.random.rand()*100 - 50
psi = np.random.rand()*360
theta = np.random.rand()*360
phi = 0

data = np.array([x, y, z, psi, theta, phi])
np.savetxt('startdata.txt', data)
print(data)


