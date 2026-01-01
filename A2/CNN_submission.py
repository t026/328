import timeit
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import transforms, datasets, models
import numpy as np
import random


#Function for reproducibilty. You can check out: https://pytorch.org/docs/stable/notes/randomness.html
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(100)

#TODO: Populate the dictionary with your hyperparameters for training
def get_config_dict(pretrain):
    """
    pretrain: 0 or 1. Can be used if you need different configs for part 1 and 2.
    """
    
    config = {
        "batch_size": 400,
        "lr": 0.001,
        "num_epochs": 10,
        "weight_decay": 0.0001,   #set to 0 if you do not want L2 regularization
        "save_criteria": 'accuracy',     #Str. Can be 'accuracy'/'loss'/'last'. (Only for part 2)

    }
    
    return config
    

#TODO: Part 1 - Complete this with your CNN architecture. Make sure to complete the architecture requirements.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        #layer 1
        x = self.pool(F.relu(self.conv1(x)))
        #;auer 2
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        #layer 3
        x = F.relu(self.fc1(x))
        #layer 4
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#TODO: Part 2 - Complete this with your Pretrained CNN architecture. 
class PretrainedNet(nn.Module):
    def __init__(self):
        super(PretrainedNet, self).__init__()
        # TODO: Load a pretrained model
        self.model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)

        print(self.model)
    def forward(self, x):
        x = self.model.forward(x)
        return x

#Feel free to edit this with your custom train/validation splits, transformations and augmentations for CIFAR-10, if needed.
def load_dataset(pretrain):
    """
    pretrain: 0 or 1. Can be used if you need to define different dataset splits/transformations/augmentations for part 2.

    returns:
    train_dataset, valid_dataset: Dataset for training your model
    test_transforms: Default is None. Edit if you would like transformations applied to the test set. 

    """

    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    train_dataset, valid_dataset = random_split(full_dataset, [38000, 12000])

    test_transforms = None

    
    return train_dataset, valid_dataset, test_transforms



