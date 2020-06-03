import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision

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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
epochs = 1000
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(my_nn.parameters(), lr = 0.001)
for epoch in range(5):  # loop over the dataset multiple times

    #running_loss = 0.0
    for i, (inputs, labels) in enumerate(data_train):
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = my_nn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0 and i != 0:
            print(i, loss.item())


my_nn.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in data_test:
        images = images.to(device)
        labels = labels.to(device)
        outputs = my_nn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
result = my_nn(random_data)
print(result)