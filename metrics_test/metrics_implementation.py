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
class Net(nn.Module):
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

def generate_data():
    x = np.arange(-5, 5, 0.1)
    y = 7*np.sin(x) + 3*abs(np.cos(x/2))*np.random.randn(len(x))
    return x, y


def normalize_it(y):
    y = y + abs(np.min(y))
    y_norm = y/np.sum(y)
    return y_norm


def load_data(nr):
    x, y = np.loadtxt("./data/paper_heteroscedastic_data_test_{}.txt".format(nr))
    return x,y


def kullback_leibler(x1, x2):
    D_KL = np.sum(x1 * np.log(x1/x2))
    return D_KL


def emd_distance(x1, x2):
    


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

plt.plot(x1, y1, '.b')

plt.plot(x2, y2, '.r')

net = Net()
net.load_state_dict(torch.load('./weights/heteroscedastic/3layer_epochs_1000_withdropout.pt'))
net.eval()

data_pred1, _ = generate_data()
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
plt.plot(data_pred1, normalize_it(real_y))
plt.plot(data_pred2, normalize_it(predictions2), '.b', label = "Predicted Data")
plt.savefig("./plots/metrics/comparison_pred_real.png")
plt.show()

plt.plot(data_pred2, normalize_it(y), '.b')
plt.plot(data_pred1, normalize_it(real_y))
plt.savefig("./plots/metrics/comparison_withuncertainty_real.png")
plt.show()

D_KL12 = kullback_leibler(predictions, predictions2)
print(D_KL12)

print(wasserstein_distance(normalize_it(predictions), normalize_it(real_y)))
print(wasserstein_distance(normalize_it(predictions2), normalize_it(real_y)))
print(wasserstein_distance(normalize_it(y), normalize_it(real_y)))

