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
from ekp_style import set_ekp_style
set_ekp_style(set_sizes=True, set_background=True, set_colors=True)


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
    #y = y/np.sum(y)
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
    cs_sum = 0
    for i in range(len(y1)):
        if y1[i] != 0 and y2[i] != 0:
            cs_sum += (y1[i]-y2[i])**2/(y1[i]+y2[i])
    return 0.5*cs_sum


# To compare different one-dimensional distributions, we generate data following the heteroscedastic function 
# $y = 7 \sin(x) + 3 |\cos(x/2)| \cdot \epsilon$ with $\epsilon\in\mathcal(0,1)$ .
# To try to quantify the output of the different distance measurements, the distance between the real heteroscedastic data and the output of the neural network is compared as well as the distance between the underlying functions without errors, $y = 7\sin(x)$, to the heteroscedastic data and the predicted data.  

# +
net = Net()
net.load_state_dict(torch.load('./weights/heteroscedastic/3layer_epochs_5000_withdropout.pt'))
net.eval()
repeats = 5000
kl_df = pd.DataFrame(columns = ['1sin', '2sin', '3sin', '4sin', '5sin', '6sin', '7sin'])
hd_df = pd.DataFrame(columns = ['1sin', '2sin', '3sin', '4sin', '5sin', '6sin', '7sin'])
k_df = pd.DataFrame(columns = ['1sin', '2sin', '3sin', '4sin', '5sin', '6sin', '7sin'])
tv_df = pd.DataFrame(columns = ['1sin', '2sin', '3sin', '4sin', '5sin', '6sin', '7sin'])
cs_df = pd.DataFrame(columns = ['1sin', '2sin', '3sin', '4sin', '5sin', '6sin', '7sin'])
w_df = pd.DataFrame(columns = ['1sin', '2sin', '3sin', '4sin', '5sin', '6sin', '7sin'])

#comparison between 1D histograms 1000 times to get distance measurement distribution
for j in range(1, 8):
    changekl = []
    changehd = []
    changek = []
    changetv = []
    changecs = []
    changew = []
    for i in range(repeats):
        x_dist, y_dist = generate_rand_data(500, 3, 7) #generate heteroscedastic data
        x_dist2, y_dist2 = generate_rand_data(500, 3, j) 
        changekl.append(kullback_leibler(y_dist, y_dist2))
        changehd.append(hellinger_distance(y_dist, y_dist2))
        changek.append(kolmogorov_metric(y_dist, y_dist2))
        changetv.append(total_variation(y_dist, y_dist2))
        changecs.append(chi_squared(y_dist, y_dist2))
        changew.append(wasserstein_distance(y_dist, y_dist2))
    kl_df['{}sin'.format(j)] = changekl
    hd_df['{}sin'.format(j)] = changehd
    k_df['{}sin'.format(j)] = changek
    tv_df['{}sin'.format(j)] = changetv
    cs_df['{}sin'.format(j)] = changecs
    w_df['{}sin'.format(j)] = changew
kl_df.to_csv("./data/metrics/Kullback_Leibler_{}repeats_range1_7.csv".format(repeats), index=False)
hd_df.to_csv("./data/metrics/Hellinger_distance_{}repeats_range1_7.csv".format(repeats), index=False)
k_df.to_csv("./data/metrics/Kolmogorov_{}repeats_range1_7.csv".format(repeats), index=False)
tv_df.to_csv("./data/metrics/Total_Variation_{}repeats_range1_7.csv".format(repeats), index=False)
cs_df.to_csv("./data/metrics/Chi_squared_{}repeats_range1_7.csv".format(repeats), index=False)
w_df.to_csv("./data/metrics/Wasserstein_{}repeats_range1_7.csv".format(repeats), index=False)


    #predictions = []
    #data_pred = np.transpose(np.vstack((x_func, np.random.randn(len(x_func)))))
    #data_pred_torch = torch.from_numpy(data_pred).float()
    #for data in data_pred_torch:
    #    predictions.append(test_data(data, net)) #generate predictions of the network
    #predictions = np.asarray(predictions)
    #predictions = normalize_it(predictions)
# -

plt.figure(figsize=(10,7))
for j in (1, 3, 5):
    mean = np.round(np.mean(np.asarray(kl_df['{}sin'.format(j)])), 3)
    plt.hist(kl_df['{}sin'.format(j)], label='{}: {}'.format(j, mean))
plt.title("Kullback-Leibler distance")
plt.legend()
plt.savefig('./plots/metrics/1D/kullbackleibler_{}repeats_change1_3_5.pdf'.format(repeats))
plt.show()
plt.figure(figsize=(10,7))
for j in (1, 3, 5):
    mean = np.round(np.mean(np.asarray(hd_df['{}sin'.format(j)])), 3)               
    plt.hist(hd_df['{}sin'.format(j)], label='{}: {}'.format(j, mean))
plt.title("Hellinger distance")
plt.legend()
plt.savefig('./plots/metrics/1D/hellingerdistance_{}repeats_change1_3_5.pdf'.format(repeats))
plt.show()
plt.figure(figsize=(10,7))
for j in (1, 3, 5):
    mean = np.round(np.mean(np.asarray(k_df['{}sin'.format(j)])), 3)
    plt.hist(k_df['{}sin'.format(j)], label='{}: {}'.format(j, mean))
plt.title("Kolmogorov distance")
plt.legend()
plt.savefig('./plots/metrics/1D/kolmogorov_{}repeats_change1_3_5.pdf'.format(repeats))
plt.show()
plt.figure(figsize=(10,7))
for j in (1, 3, 5):
    mean = np.round(np.mean(np.asarray(tv_df['{}sin'.format(j)])), 3)
    plt.hist(tv_df['{}sin'.format(j)], label='{}: {}'.format(j, mean))
plt.title("Total variation distance")
plt.legend()
plt.savefig('./plots/metrics/1D/totalvariation_{}repeats_change1_3_5.pdf'.format(repeats))
plt.show()
plt.figure(figsize=(10,7))
for j in (1, 3, 5):
    mean = np.round(np.mean(np.asarray(cs_df['{}sin'.format(j)])), 3)
    plt.hist(cs_df['{}sin'.format(j)], label='{}: {}'.format(j, mean))
plt.title("Chi-squared distance")
plt.legend()
plt.savefig('./plots/metrics/1D/chisquared_{}repeats_change1_3_5.pdf'.format(repeats))
plt.show()
plt.figure(figsize=(10,7))
for j in (1, 3, 5):
    mean = np.round(np.mean(np.asarray(w_df['{}sin'.format(j)])), 3)
    plt.hist(w_df['{}sin'.format(j)], label='{}: {}'.format(j, mean))
plt.title("Wasserstein distance")
plt.legend()
plt.savefig('./plots/metrics/1D/wasserstein_{}repeats_change1_3_5.pdf'.format(repeats))
plt.show()
