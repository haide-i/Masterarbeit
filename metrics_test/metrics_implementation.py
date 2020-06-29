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

# +
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wasserstein_distance
import torch
import torchvision 
import torchvision.transforms as transforms
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# +
class Net(nn.Module): #load trained NN to generate heteroscedastic data
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 30)
        self.drop = nn.Dropout(0.5)
        self.fc3 = nn.Linear(30, 20)
        self.drop2 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.drop2(x)
        x = self.fc4(x)
        return x
    
def test_data(val_data, net):
    with torch.no_grad():
        data_pred = net(val_data.float())
    return data_pred


# -

def generate_data(length):
    x = np.linspace(-5, 5, length)
    y = 7*np.sin(x) + 3*abs(np.cos(x/2))*np.random.randn(len(x))
    y = y + abs(np.min(y))
    y = y/np.sum(y)
    return x, y


def generate_func_data(length): #generate data with random distributed x-values but y-values without errors
    x = np.linspace(-5, 5, 1000)
    random_ints = np.random.randint(0, len(x), length)
    x = x[random_ints]
    y = 7*np.sin(x)
    y = y + abs(np.min(y))
    y = y/np.sum(y)
    return x, y


def generate_rand_data(length, diff, change): #generate data with random distributed x-values and heteroscedastic y-values
    x = np.linspace(-5, 5, 1000)
    random_ints = np.random.randint(0, len(x), length)
    x = x[random_ints]
    y = change*np.sin(x) + diff*abs(np.cos(x/2))*np.random.randn(len(x))
    y = y + abs(np.min(y))
    y = y/np.sum(y)
    return x, y


def normalize_it(y):
    y = y + abs(np.min(y))
    y_norm = y/np.sum(y)
    return y_norm


def load_data(nr):
    x, y = np.loadtxt("./data/paper_heteroscedastic_data_test_{}.txt".format(nr))
    return x,y


# # Metrics
# Implementation of different metrics to compare one-dimensional distributions. Not all distance measurements used here are metrics, as for example the Kullback-Leibler divergence is not symmetric. In case of the KL divergence all points with measurement 0 are not taken into account. Distance measurements used here are the KL divergence, the Hellinger distance, The Kolmogorov-Smirnov metric, the total variation distance, the $\chi^2$-distance and the Wasserstein metric (from scipy).

def kullback_leibler(x1, x2):
    D_KL= 0
    for i in range(len(x1)):
        if x1[i] != 0 and x2[i] != 0:
            D_KL += x1[i] * np.log(x1[i]/x2[i])
    return D_KL


def hellinger_distance(y1, y2):
    return np.sqrt(np.sum((np.sqrt(y1) - np.sqrt(y2)) ** 2)) / np.sqrt(2)


def kolmogorov_metric(y1, y2):
    return np.max(np.cumsum(y1)-np.cumsum(y2))


def separation_distance(y1, y2):
    
    return(np.max(1-y1/y2))


def total_variation(y1, y2):
    return 0.5*np.sum(abs(y1-y2))


def chi_squared(y1, y2):
    return 0.5*np.sum((y1-y2)**2/(y1+y2))


# To compare different one-dimensional distributions, we generate data following the heteroscedastic function 
# $y = 7 \sin(x) + 3 |\cos(x/2)| \cdot \epsilon$ with $\epsilon\in\mathcal(0,1)$ .
# To try to quantify the output of the different distance measurements, the distance between the real heteroscedastic data and the output of the neural network is compared as well as the distance between the underlying functions without errors, $y = 7\sin(x)$, to the heteroscedastic data and the predicted data.  

# +
net = Net()
net.load_state_dict(torch.load('./weights/heteroscedastic/3layer_epochs_5000_withdropout.pt'))
net.eval()
kl_distance = []
hellinger = []
kolmogorov = []
totvar = []
chisquare = []
wasserstein = []
kl_distance_real = []
hellinger_real = []
kolmogorov_real = []
totvar_real = []
chisquare_real = []
wasserstein_real = []
#comparison between 1D histograms 1000 times to get distance measurement distribution
for i in range(5000):
    x_dist, y_dist = generate_rand_data(500, 3, 7) #generate heteroscedastic data
    x_func, y_func = generate_rand_data(500, 3, 4) 
    x_dist2, y_dist2 = generate_rand_data(500, 3, 7)
    #predictions = []
    #data_pred = np.transpose(np.vstack((x_func, np.random.randn(len(x_func)))))
    #data_pred_torch = torch.from_numpy(data_pred).float()
    #for data in data_pred_torch:
    #    predictions.append(test_data(data, net)) #generate predictions of the network
    #predictions = np.asarray(predictions)
    #predictions = normalize_it(predictions)
    kl_distance_real.append(kullback_leibler(y_func, y_dist))
    hellinger_real.append(hellinger_distance(y_func, y_dist))
    kolmogorov_real.append(kolmogorov_metric(y_func, y_dist))
    totvar_real.append(total_variation(y_func, y_dist))
    chisquare_real.append(chi_squared(y_func, y_dist))
    wasserstein_real.append(wasserstein_distance(y_func, y_dist))
    kl_distance.append(kullback_leibler(y_dist2, y_dist))
    hellinger.append(hellinger_distance(y_dist2, y_dist))
    kolmogorov.append(kolmogorov_metric(y_dist2, y_dist))
    totvar.append(total_variation(y_dist2, y_dist))
    chisquare.append(chi_squared(y_dist2, y_dist))
    wasserstein.append(wasserstein_distance(y_dist2, y_dist))
    
df = pd.DataFrame({"Kullback-Leibler" : kl_distance, "Hellinger" : hellinger, "Kolmogorov" : kolmogorov, 
                   "Total variation" : totvar, "Chi-squared" : chisquare,
                  "Wasserstein" : wasserstein})
df.to_csv("./data/metrics/1Dmetrics_samedist_5000repeats.csv", index=False)
df2 = pd.DataFrame({"Kullback-Leibler" : kl_distance_real, "Hellinger" : hellinger_real, 
                    "Kolmogorov" : kolmogorov_real, "Total variation" : totvar_real, "Chi-squared" : chisquare_real,
                  "Wasserstein" : wasserstein_real})
df.to_csv("./data/metrics/1Dmetrics_change47_5000repeats.csv", index=False)
plt.hist(kl_distance, bins=20, alpha = 0.5, label = "Function - Prediction")
plt.hist(kl_distance_real, bins=20, alpha = 0.5, label = "7sin(x) - 4sin(x)")
plt.title("Kullback-Leibler")
plt.legend()
plt.savefig("./plots/metrics/1D/kullbackleibler_change47.png")
plt.show()
plt.hist(hellinger, bins=20, alpha = 0.5, label = "Same distribution")
plt.hist(hellinger_real, bins=20, alpha = 0.5, label = "7sin(x) - 4sin(x)")
plt.title("Hellinger")
plt.legend()
plt.savefig("./plots/metrics/1D/hellinger_change47.png")
plt.show()
plt.hist(kolmogorov, bins=20, alpha = 0.5, label = "Same distribution")
plt.hist(kolmogorov_real, bins=20, alpha = 0.5, label = "7sin(x) - 4sin(x)")
plt.title("Kolmogorov")
plt.legend()
plt.savefig("./plots/metrics/1D/kolmogorov_change47.png")
plt.show()
plt.hist(totvar, bins=20, alpha = 0.5, label = "Same distribution")
plt.hist(totvar_real, bins=20, alpha = 0.5, label = "7sin(x) - 4sin(x)")
plt.title("Total variation")
plt.legend()
plt.savefig("./plots/metrics/1D/totvar_change47.png")
plt.show()
plt.hist(chisquare, bins=20, alpha = 0.5, label = "Same distribution")
plt.hist(chisquare_real, bins=20, alpha = 0.5, label = "7sin(x) - 4sin(x)")
plt.title("Chi-squared")
plt.legend()
plt.savefig("./plots/metrics/1D/chisquared_change47.png")
plt.show()
plt.hist(wasserstein, bins=20, alpha = 0.5, label = "Same distributionn")
plt.hist(wasserstein_real, bins=20, alpha = 0.5, label = "7sin(x) - 4sin(x)")
plt.title("Wasserstein")
plt.legend()
plt.savefig("./plots/metrics/1D/wasserstein_change47.png")
plt.show()
# -

x1, y1 = load_data(0)
y1 = normalize_it(y1)
print(np.min(y1))
x2, y2 = load_data(0)
y2 = normalize_it(y2)
x3, y3 = load_data(0)
y3 = normalize_it(y3)
x4, y4 = load_data(0)
y4 = normalize_it(y4)

real_x = np.arange(-5, 5, 0.1)
real_y =  7*np.sin(real_x) + 3*abs(np.cos(real_x/2))
x_func, y_func = generate_rand_data(500, 3, 1) 
x_dist2, y_dist2 = generate_rand_data(500, 3, 7)
plt.plot(x_func, y_func, '.b')
plt.plot(x_dist2, y_dist2, '.r')
plt.show()

net = Net()
net.load_state_dict(torch.load('./weights/heteroscedastic/3layer_epochs_5000_withdropout.pt'))
net.eval()

data_pred1, test = generate_data()
data_pred = np.transpose(np.vstack((data_pred1, np.random.randn(len(data_pred1)))))
data_pred_torch = torch.from_numpy(data_pred).float()
predictions = []
for data in data_pred_torch:
    predictions.append(test_data(data, net))
predictions = np.asarray(predictions)
plt.plot(data_pred1, normalize_it(real_y))
plt.plot(data_pred1, normalize_it(predictions), '.b', label = "Predicted Data")
plt.show()

data_pred2, y = generate_data()
data_pred = np.transpose(np.vstack((data_pred2, np.random.randn(len(data_pred2)))))
data_pred_torch = torch.from_numpy(data_pred).float()
predictions2 = []
for data in data_pred_torch:
    predictions2.append(test_data(data, net))
predictions2 = np.asarray(predictions2)
plt.plot(data_pred1, normalize_it(real_y), 'r', label='Function')
plt.plot(data_pred2, normalize_it(predictions2), '.b', label = "Predicted Data")
#plt.plot(data_pred2, normalize_it(y), '+g', label = 'With uncertainty')
plt.legend(loc='upper right')
plt.savefig("./plots/metrics/comparison_pred_real.png")
plt.show()

plt.plot(data_pred2, normalize_it(y), '.b', label = 'With uncertainty')
plt.plot(data_pred1, normalize_it(real_y), 'r', label='Function')
plt.legend(loc='upper right')
plt.savefig("./plots/metrics/comparison_withuncertainty_real.png")
plt.show()
