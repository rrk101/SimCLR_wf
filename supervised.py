import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torchvision.models.resnet import resnet50
import torch.nn.functional as F

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=16)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)


# Base model definition
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=False, num_classes=10)

    def forward(self, x):
        x = self.resnet50(x)
        return x

class Supervised_Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, 10, bias=True))
        

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(out, dim=-1)

# Supervised ResNet50 model training
def train_supervised(model, trainloader, criterion, optimizer, epochs):
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    net = Supervised_Model().cuda()
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    losses=[]
    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        pbar = tqdm(trainloader)
        i = 0

        for data in pbar:
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
                
            if i%100==0:
                losses.append(loss.data.numpy())
                print(losses)
                pass

            pbar.set_description("Processing epoch {:d} minibatch {:d} train loss {:.3f}".format(epoch,\
                                                                    i+1, running_loss/(i+1)))
            i += 1      
    print(losses)

    print('Finished Training')



def test_supervised(model, tloader, criterion, optimizer, epochs):
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    net = Supervised_Model().cuda()
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    losses=[]
    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        pbar = tqdm(trainloader)
        i = 0

        for data in pbar:
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
                
            if i%100==0:
                losses.append(loss.data.numpy())
                print(losses)
                pass

            pbar.set_description("Processing epoch {:d} minibatch {:d} train loss {:.3f}".format(epoch,\
                                                                    i+1, running_loss/(i+1)))
            i += 1      
    print(losses)

    print('Finished Training')
    # device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # for epoch in range(epochs):
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         inputs, labels = data
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    #     print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))


# Testing both models on test set
# def test(model, testloader):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     accuracy = 



import torch
import torchvision
import numpy as np

def get_percentage_class_balanced_data(percentage=0.01, batch_size= 128):
    """
    returns dataloader for cifar-10 'percentage' of training data(class balanced)
    """
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=utils.train_transform_org)
    # Get the class labels and counts
    classes, counts = np.unique(trainset.targets, return_counts=True)

    # Calculate the number of samples to get per class
    num_samples_per_class = int(percentage * np.min(counts))

    # Create a dictionary to store the indices of the samples for each class
    class_indices = {label: [] for label in classes}

    # Loop through the training data and add the indices of each sample to the dictionary
    for i in range(len(trainset)):
        image, label = trainset[i]
        class_indices[label].append(i)

    # Create a list to store the indices of the selected samples
    selected_indices = []

    # Loop through the classes and randomly select the desired number of samples from each class
    for label in classes:
        indices = class_indices[label]
        selected_indices.extend(np.random.choice(indices, num_samples_per_class, replace=False))

    # Create a Subset of the training data using the selected indices
    subset = torch.utils.data.Subset(trainset, selected_indices)

    # Create a DataLoader for the Subset
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
    return subset_loader


# # Define the batch size for loading the data
# batch_size = 64

# # Define the transformation to apply to the data
# transform = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# # Load the CIFAR-10 training data
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)

# # Get the class labels and counts
# classes, counts = np.unique(trainset.targets, return_counts=True)

# # Calculate the number of samples to get per class
# num_samples_per_class = int(0.01 * np.min(counts))

# # Create a dictionary to store the indices of the samples for each class
# class_indices = {label: [] for label in classes}

# # Loop through the training data and add the indices of each sample to the dictionary
# for i in range(len(trainset)):
#     image, label = trainset[i]
#     class_indices[label].append(i)

# # Create a list to store the indices of the selected samples
# selected_indices = []

# # Loop through the classes and randomly select the desired number of samples from each class
# for label in classes:
#     indices = class_indices[label]
#     selected_indices.extend(np.random.choice(indices, num_samples_per_class, replace=False))

# # Create a Subset of the training data using the selected indices
# subset = torch.utils.data.Subset(trainset, selected_indices)

# # Create a DataLoader for the Subset
# subset_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
