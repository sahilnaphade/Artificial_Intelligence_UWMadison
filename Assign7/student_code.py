# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms

class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # Layer 1 (CN1)
        self.convLayer1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU()
        self.maxPoolLayer1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2 (CN2)
        self.convLayer2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.maxPoolLayer2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        
        # Layer 3: Flattening layer
        self.flatten1 = nn.Flatten()
        
        # Layer 4: Linear layer 1 (16*5*5 -> 256) -- we have a kernel size of 5 with stride=1
        self.linear1 = nn.Linear(16*5*5, 256)
        self.relu3 = nn.ReLU()
        
        # Layer 5: Linear layer 2 (256 -> 128)
        self.linear2 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()
        
        # Layer 6: Output layer (128 -> 100)
        self.outputLayer = nn.Linear(128, num_classes)
        

    def forward(self, x):
        shape_dict = {}
        # 1. Run layer 1
        x = self.relu1(self.convLayer1(x))
        x = self.maxPoolLayer1(x)
        shape_dict[1] = list(x.size())

        # 2. Run layer 2
        x = self.relu2(self.convLayer2(x))
        x = self.maxPoolLayer2(x)
        shape_dict[2] = list(x.size())

        # 3. Flatten
        x = self.flatten1(x)
        shape_dict[3] = list(x.size())

        # 4. Linear 1
        x = self.relu3(self.linear1(x))
        shape_dict[4] = list(x.size())

        # 5. Linear 2
        x = self.relu4(self.linear2(x))
        shape_dict[5] = list(x.size())

        # 6. Output layer
        out = self.outputLayer(x)
        output_s = out.shape
        shape_dict[6] = list(x.size())
        

        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    total_count_params = 0
    model = LeNet()
    named_params = model.named_parameters()
    for name, param in named_params:
        if not param.requires_grad:
            continue
        total_count_params += param.numel()

    return total_count_params / 1e6


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
