import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision import transforms
import torchvision
from skimage.io import imread
from PIL import Image
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import math


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64 * 2, 32, 3, 1)
        self.upconv1 = self.expand_block(32 * 2, out_channels, 3, 1)

    def forward(self, x):
        # Downsampling
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # Upsampling
        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        
class MNISTDDDataset(Dataset):
    def __init__(self, image_paths, target_paths, train=True):   # initial logic happens like transform

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transforms_image = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    def __getitem__(self, index):
        image = self.image_paths[index]
        image = image.reshape(64, 64, 3)
        mask = (self.target_paths[index])
        t_image = self.transforms_image(image)
        return t_image, mask

    def __len__(self):  # return count of sample we have

        return len(self.image_paths)




def main():
    train = np.load("train.npz")
    train_dataset = MNISTDDDataset(train["images"], train["semantic_masks"])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    network = UNET(3, 11).to(device)
    optimizer = optim.Adam(network.parameters(), lr=1e-3)

    for itr in range(1000):
        loss_val=0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = network(data)
            #print(output.shape,target.shape)
            loss = F.nll_loss(F.log_softmax(output,dim=1), target)
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
        if itr%100==0:
            print(loss_val)
if __name__ == "__main__":
    main()
