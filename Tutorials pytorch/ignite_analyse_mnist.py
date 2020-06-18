import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
from ignite.handlers import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from torch.utils.tensorboard import SummaryWriter

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output

def score_function(engine):
    val_loss = engine.state.metrics['loss']
    return -val_loss

def log_training_results(trainer):
    train_evaluator.run(data_train)
    metrics = train_evaluator.state.metrics
    accuracy = metrics['accuracy']*100
    loss = metrics['loss']
    last_epoch.append(0)
    training_history['accuracy'].append(accuracy)
    training_history['loss'].append(loss)
    print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, accuracy, loss))

def log_validation_results(trainer):
    val_evaluator.run(data_test)
    metrics = val_evaluator.state.metrics
    accuracy = metrics['accuracy']*100
    loss = metrics['loss']
    validation_history['accuracy'].append(accuracy)
    validation_history['loss'].append(loss)
    print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, accuracy, loss))
    
def log_confusion_matrix(trainer):
    val_evaluator.run(data_test)
    metrics = val_evaluator.state.metrics
    cm = metrics['cm']
    cm = cm.numpy()
    cm = cm.astype(int)
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    fig, ax = plt.subplots(figsize=(10,10))  
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax,fmt="d")
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels') 
    ax.set_title('Confusion Matrix') 
    ax.xaxis.set_ticklabels(classes,rotation=90)
    ax.yaxis.set_ticklabels(classes,rotation=0)
    plt.show()

def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]
    
    writer.add_pr_curve(classes[class_index], tensorboard_preds, tensorboard_probs, global_step=global_step)
    writer.close()


classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

device = torch.device('cpu')
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])
random_data = torch.rand((1,1,28,28))
mnist_train = torchvision.datasets.MNIST('./', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST('./', train=False, download=True, transform=transforms.ToTensor())

data_train = torch.utils.data.DataLoader(mnist_train,
                                          batch_size=32,
                                          shuffle=True,
                                          )
data_test = torch.utils.data.DataLoader(mnist_test,
                                          batch_size=32,
                                          shuffle=False,
                                        )
my_nn=Net()
epochs = 5
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(my_nn.parameters(), lr = 0.001)

writer = SummaryWriter('MNIST')
trainer = create_supervised_trainer(my_nn, optimizer, criterion, device=device)
metrics = {'accuracy':Accuracy(), 'loss':Loss(criterion), 'cm':ConfusionMatrix(num_classes=10)}

train_evaluator = create_supervised_evaluator(my_nn, metrics=metrics, device=device)
val_evaluator = create_supervised_evaluator(my_nn, metrics=metrics, device=device)

training_history = {'accuracy':[], 'loss':[]}
validation_history = {'accuracy': [], 'loss':[]}
last_epoch = []

RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

handler = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
val_evaluator.add_event_handler(Events.COMPLETED, handler)

trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)
trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

trainer.add_event_handler(Events.COMPLETED, log_confusion_matrix)

trainer.run(data_train, max_epochs=epochs)

class_probs = []
class_preds = []
with torch.no_grad():
    for data in data_test:
        images, labels = data
        output = my_nn(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]
        _, class_preds_batch = torch.max(output, 1)
        
        class_probs.append(class_probs_batch)
        class_preds.append(class_preds_batch)
test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_preds = torch.cat(class_preds)
for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, test_preds)

plt.plot(training_history['accuracy'],label="Training Accuracy")
plt.plot(validation_history['accuracy'],label="Validation Accuracy")
plt.xlabel('No. of Epochs')
plt.ylabel('Accuracy')
plt.legend(frameon=False)
plt.show()
plt.plot(training_history['loss'],label="Training Loss")
plt.plot(validation_history['loss'],label="Validation Loss")
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.legend(frameon=False)
plt.show()
