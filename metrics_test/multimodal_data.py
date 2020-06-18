import numpy as np
import matplotlib.pyplot as plt
import math

#x = np.arange(0, 100, 0.5)
nr_data = 150
x = np.zeros(nr_data)
y = np.zeros(len(x))
print(len(x))
for j in range(4):
    for i in range(nr_data):
        x[i] = 2.5*np.random.rand() - 0.5
        if np.random.rand() >= 0.5:
            y[i] = 10*math.sin(x[i]) + np.random.randn()
        else:    
            y[i] = 10*math.cos(x[i]) + np.random.randn()

    np.savetxt("./data/paper_multimodal_data_test_sparse_{}.txt".format(j), (x, y))

#plt.plot(x, y, '.b')
#plt.show()


