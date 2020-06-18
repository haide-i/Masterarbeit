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
import matplotlib.pyplot as plt

import torch
import torchvision 
import torchvision.transforms as transforms
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping


# -

def generate_data(length):
    x = 2.5*np.random.rand(length) - 0.5
    rand_sep = int(np.random.rand()*length)
    y = np.empty(length)
    y[:rand_sep] = 10*np.sin(x[:rand_sep]) + np.random.randn(rand_sep)
    y[rand_sep:] = 10*np.cos(x[rand_sep:]) + np.random.randn(length - rand_sep)
    return x, y
    


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 30)
        self.fc2 = nn.Linear(30, 40)
        self.drop1 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(40, 20)
        self.drop2 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop1(x)
        x = F.relu(self.fc3(x))
        x = self.drop2(x)
        x = self.fc4(x)
        return x


# +
def log_training_results(trainer):
    train_evaluator.run(data_train)
    metrics = train_evaluator.state.metrics
    loss = metrics['loss']
    last_epoch.append(0)
    training_history['loss'].append(loss)
    print("Training Results - Epoch: {}   Avg loss: {:.2f}"
          .format(trainer.state.epoch, loss))
    
def log_validation_results(trainer):
    val_evaluator.run(data_val)
    metrics = val_evaluator.state.metrics
    loss = metrics["loss"] 
    validation_history["loss"].append(loss)
    print("Validation Results - Epoch: {}  Avg loss: {:.2f}"
          .format(trainer.state.epoch, loss))
    
def test_data(val_data, net):
    with torch.no_grad():
        data_pred = net(data.float())
    return data_pred


# -

length = 750
x, y = generate_data(length)
x_val, y_val = generate_data(length)

# +
device = torch.device("cpu")

x_train = np.transpose(np.vstack((x, np.random.randn(len(x)))))
x_eval = np.transpose(np.vstack((x_val, np.random.randn(len(x_val)))))

x_train = torch.from_numpy(x_train).float().to(device)
y_train = torch.from_numpy(y).float().to(device)
x_val = torch.from_numpy(x_eval).float().to(device)
y_val = torch.from_numpy(y_val).float().to(device)

dataset = TensorDataset(x_train, y_train)
dataset_val = TensorDataset(x_val, y_val)
data_train = DataLoader(dataset, batch_size=1, shuffle=True)
data_val = DataLoader(dataset_val, batch_size=1, shuffle=False)

# +
net = Net()
epochs = 20
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

trainer = create_supervised_trainer(net, optimizer, criterion, device=device)
metrics = {'loss':Loss(criterion)}

train_evaluator = create_supervised_evaluator(net, metrics=metrics, device=device)
val_evaluator = create_supervised_evaluator(net, metrics=metrics, device=device)

training_history = {'accuracy':[], 'loss':[]}
validation_history = {'accuracy': [], 'loss':[]}
last_epoch = []

RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

# +
trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)
trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)
net = net.float()

trainer.run(data_train, max_epochs=epochs)

torch.save(net.state_dict(), "./weights/multimodal/3layer_epochs_{}_withdropout.pt".format(epochs))
# -

plt.plot(training_history['loss'],label="Training Loss")
plt.plot(validation_history['loss'],label="Validation Loss")
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.legend(frameon=False)
plt.savefig("./plots/multimodal/loss_function_3layer_noepochs_{}_withdropout.png".format(epochs))
plt.show()

test_length = 750
data_pred, _ = generate_data(test_length)
data_pred = np.transpose(np.vstack((data_pred, np.random.randn(len(data_pred)))))
data_pred_torch = torch.from_numpy(data_pred).float()
predictions = []
for data in data_pred_torch:
    predictions.append(test_data(data, net))
plt.plot(data_pred, predictions, '.b', label = "Predicted Data")
plt.savefig("./plots/multimodal/pred_pts{}_nrepochs_{}_3layer_withdropout.png".format(test_length, epochs))
plt.show()


