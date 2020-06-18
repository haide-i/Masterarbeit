import numpy as np
import matplotlib.pyplot as plt
import csv
import math

def normalize_it(y):
    y = y + abs(np.min(y))
    y_norm = y/np.sum(y)
    return y_norm

def generate_norm_data(length, diff):
    x = np.zeros(length)
    y = np.zeros(len(x))
    for i in range(length):
        if i < length/3:
            x[i] = 2/5*np.random.randn()-4 
        elif i >= length/3 and i < length*2/3:
            x[i] = 0.9*np.random.randn() 
        else:    
            x[i] = 2/5*np.random.randn()+4

    y = 7*np.sin(x) + 3*abs(np.cos(x/2))*np.random.randn(length)

    y = y + abs(np.min(y))
    y_norm = y/np.sum(y)
    return x, y_norm


def generate_data():
    x = np.arange(-5, 5, 0.1)
    y = 7*np.sin(x) + 3*abs(np.cos(x/2))*np.random.randn(len(x))
    return x, y


x, y = generate_norm_data(1000, 3)
real_x = np.arange(-5, 5, 0.01)
print(len(x), len(real_x))
real_y =  7*np.sin(real_x) #+ 3*abs(np.cos(real_x/2))#*np.random.randn(len(real_x))
plt.plot(real_x, normalize_it(real_y))

# +
plt.plot(x, y, '.b', label="With uncertainty")

#plt.plot(x, normalize_it(y), '.b')

plt.plot(real_x, normalize_it(real_y), 'r', label="Function")
plt.legend(loc='upper right')
plt.savefig("./plots/metrics/realdata_with_function.png")
# -

plt.show()


