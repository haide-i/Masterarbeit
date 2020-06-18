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
    
def score_function(engine):
    val_loss = engine.state.metrics['loss']
    return -val_loss
    
def log_training_results(trainer):
    train_evaluator.run(data_train)
    metrics = train_evaluator.state.metrics
    #accuracy = metrics['accuracy']*100
    loss = metrics['loss']
    last_epoch.append(0)
    #training_history['accuracy'].append(accuracy)
    training_history['loss'].append(loss)
    print("Training Results - Epoch: {}   Avg loss: {:.2f}"
          .format(trainer.state.epoch, loss))
    
def log_validation_results(trainer):
    val_evaluator.run(data_val)
    metrics = val_evaluator.state.metrics
    #accuracy = metrics["accuracy"]*100
    loss = metrics["loss"]
    #validation_history["accuracy"].append(accuracy)
    validation_history["loss"].append(loss)
    print("Validation Results - Epoch: {}  Avg loss: {:.2f}"
          .format(trainer.state.epoch, loss))
    
def latent_variables(x):
    add_var = np.zeros((1,2))
    add_var[0][0] = x
    add_var[0][1] = np.random.randn()
    return add_var

def test_data(val_data, net):
    with torch.no_grad():
        data_pred = net(val_data.float())
    return data_pred
    
    
x, y = np.loadtxt("./data/paper_heteroscedastic_data.txt")
x_val1, y_val1 = np.loadtxt("./data/paper_heteroscedastic_data_test.txt")

device = torch.device("cpu")
x_train1 = np.zeros((1,2))
x_eval = np.zeros((1,2))
for data in x:
    x_train1 = np.append(x_train1, latent_variables(data), 0)
    
for data in x_val1:
    x_eval = np.append(x_eval, latent_variables(data), 0)
    
x_train1 = np.delete(x_train1, 0, 0)
x_eval = np.delete(x_eval, 0, 0)

x_train = torch.from_numpy(x_train1).float().to(device)
y_train = torch.from_numpy(y).float().to(device)
x_val = torch.from_numpy(x_eval).float().to(device)
y_val = torch.from_numpy(y_val1).float().to(device)

dataset = TensorDataset(x_train, y_train)
dataset_val = TensorDataset(x_val, y_val)
data_train = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
data_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)


net = Net()
epochs = 10
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

#handler = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
#val_evaluator.add_event_handler(Events.COMPLETED, handler)

trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)
trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)
net = net.float()

trainer.run(data_train, max_epochs=epochs)

torch.save(net.state_dict(), "./weights/3layer_epochs_{}_withdropout.pt".format(epochs))

plt.plot(training_history['loss'],label="Training Loss")
plt.plot(validation_history['loss'],label="Validation Loss")
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.legend(frameon=False)
plt.savefig("./plots/heteroscedastic_loss_function_3layer_noepochs_{}_withdropout.png".format(epochs))
plt.show()


fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
axs = axs.ravel()

for j in range(4):
    data_pred = []
    help_arr = np.zeros((1,2))
    val_data, val_labels = np.loadtxt("./data/paper_heteroscedastic_data_test_{}.txt".format(j))
    for data in val_data:
        help_arr = np.append(help_arr, latent_variables(data), 0)
    help_arr = np.delete(help_arr, 0, 0)
    val_data_torch = torch.from_numpy(help_arr)
    for data in val_data_torch:
        data_pred = np.append(data_pred, test_data(data, net))
    axs[j].plot(val_data, data_pred, '.b', label = "Predicted Data")

plt.savefig("./plots/heteroscedastic_prediction_full_no_epochs_{}_3layer_withdropout.png".format(epochs))
plt.show()
plt.close()
fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
axs = axs.ravel()

for j in range(4):
    data_pred = []
    help_arr = np.zeros((1,2))
    val_data, val_labels = np.loadtxt("./data/paper_heteroscedastic_data_test_sparse_{}.txt".format(j))
    for data in val_data:
        help_arr = np.append(help_arr, latent_variables(data), 0)
    help_arr = np.delete(help_arr, 0, 0)
    val_data_torch = torch.from_numpy(help_arr)
    for data in val_data_torch:
        data_pred = np.append(data_pred, test_data(data, net))
    axs[j].plot(val_data, data_pred, '.b', label = "Predicted Data")

plt.savefig("./plots/heteroscedastic_prediction_sparse_no_epochs_{}_3layer_withdropout.png".format(epochs))
plt.show()
plt.close()
plt.plot(x_val1, y_val1, '.r', label = "Real Data")
plt.legend()
plt.savefig("./plots/heteroscedastic_real.png")
plt.show()
