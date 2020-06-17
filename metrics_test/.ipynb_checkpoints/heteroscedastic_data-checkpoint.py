import numpy as np
import matplotlib.pyplot as plt
import csv
import math

def normalize_it(y):
    y = y + abs(np.min(y))
    y_norm = y/np.sum(y)
    return y_norm

#x = np.arange(0, 100, 0.5)
nr_data = 750
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

   # np.savetxt("./data/paper_heteroscedastic_data_test.txt", (x, y))

real_x = np.arange(-5, 5, 0.1)
real_y =  7*np.sin(real_x) + 3*abs(np.cos(real_x/2))
plt.plot(real_x, normalize_it(real_y))

y = y + abs(np.min(y))
y = y/np.sum(y)
plt.plot(x, y, '.b')
#plt.plot(real_x, real_y)
plt.savefig("./plots/metrics/realdata_with_function.png")

plt.show()
