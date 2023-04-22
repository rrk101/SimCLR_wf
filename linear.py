import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

import utils
from model import Model


class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        self.f = Model().f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return F.normalize(out, dim=-1)



def get_percentage_class_balanced_data(percentage=0.01, batch_size= 128):
    """
    returns dataloader for cifar-10 'percentage' of training data(class balanced)
    """
    trainset = CIFAR10(root='./data', train=True,
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
    return subset
    # Create a DataLoader for the Subset
    # subset_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
    # return subset_loader








# train or test for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default='results/128_0.5_200_128_100_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--is_supervised',type=bool,default=False, help='If supervised,change the result path names')
    parser.add_argument('--subset_percentage',type=float,default=100.0,help='get subset of labelled data(class balanced)')

    args = parser.parse_args()
    is_supervised = args.is_supervised
    subset_percentage = args.subset_percentage

    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs

    if is_supervised:
        model_path = 'results/128_0.5_200_128_1_supv_model.pth'

    train_data = None
    if subset_percentage >= 100.0:
        train_data = CIFAR10(root='data', train=True, transform=utils.train_transform_org, download=True)
    else:
        train_data = get_percentage_class_balanced_data(percentage=subset_percentage)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_data = CIFAR10(root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model = Net(num_class=10, pretrained_path=model_path).cuda() #note hardcoded no. of classes, change it later
    if is_supervised:
        for param in model.f.parameters():
            param.requires_grad = False
    else:
        for param in model.f.parameters():
            param.requires_grad = True


    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = None
    if is_supervised:
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    else:
        optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    best_acc = 0.0
    cat = str(subset_percentage)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        if is_supervised:
            data_frame.to_csv('results/linear_statistics_supervised_'+cat+'.csv', index_label='epoch')
        else:
            data_frame.to_csv('results/linear_statistics_'+cat+'.csv', index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            if is_supervised:
                torch.save(model.state_dict(), 'results/supervised_linear_model_'+cat+'.pth')
            else:
                torch.save(model.state_dict(), 'results/linear_model_'+cat+'.pth')
