import numpy as np
import matplotlib.pyplot as plt
import csv
import math

#x = np.arange(0, 100, 0.5)
nr_data = 7500
x = np.zeros(nr_data)
y = np.zeros(len(x))
print(len(x))
for j in range(1):
    for i in range(nr_data):
        if i < nr_data/3:
            x[i] = 2/5*np.random.randn()-4 
        elif i >= nr_data/3 and i < nr_data*2/3:
            x[i] = 0.9*np.random.randn() 
        else:    
            x[i] = 2/5*np.random.randn()+4

        y[i] = 7*math.sin(x[i]) + 3*abs(math.cos(x[i]/2))*np.random.randn() 

    np.savetxt("./data/paper_heteroscedastic_data.txt", (x, y))
