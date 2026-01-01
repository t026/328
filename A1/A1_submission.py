"""
TODO: Finish and submit your code for logistic regression, neural network, and hyperparameter search.

"""

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def logistic_regression(device):
    # source =  https://drive.google.com/file/d/1xnd2venr-iDJBGrR7BVIiH_jOXk60maT/view?usp=sharing

    MNIST_training_set = datasets.MNIST('./MNIST_dataset/', train=True, download=True,
                                transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))]))

    MNIST_validation_set = Subset(MNIST_training_set, range(60000-12000, 60000))


    class LogisticRegression(nn.Module):
        def __init__(self):
            super(LogisticRegression, self).__init__()
            self.fc = nn.Linear(28*28, 10)
            self.n_epochs = 10
            self.batch_size_train = 200
            self.learning_rate = 0.4
            self.log_interval = 100
            self.weight_decay = 1e-4
            self.train_loader = DataLoader(MNIST_training_set,batch_size=self.batch_size_train, shuffle=True)
            self.validation_loader = DataLoader(MNIST_validation_set,batch_size=self.batch_size_train, shuffle=True)

            

        def forward(self, x):
            x = x.view(x.size(0), -1)
            y = torch.sigmoid(self.fc(x))
            return y
        
        def start_training(self, data_loader, optimizer):
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = self(data)
                loss = F.cross_entropy(output, one_hot(target, num_classes=10).float())
                loss.backward()
                optimizer.step()

        def validate(self, data_loader,dataset):
            loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in data_loader:
                    data = data.to(device)
                    target = target.to(device)
                    output = self(data)
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
                    loss += F.cross_entropy(output, one_hot(target, num_classes=10).float()).item()
            loss /= len(data_loader.dataset)
            print(dataset+'set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
            return 100. * correct / len(data_loader.dataset)
    
    logisticRegression = LogisticRegression().to(device)

    optimizer = optim.SGD(logisticRegression.parameters(), lr = logisticRegression.learning_rate, weight_decay=logisticRegression.weight_decay)
    
    one_hot = torch.nn.functional.one_hot

    logisticRegression.validate(logisticRegression.validation_loader,"Validation")
    for epoch in range(1, logisticRegression.n_epochs + 1):
      logisticRegression.start_training(logisticRegression.train_loader,optimizer)
      if not epoch%2:
        logisticRegression.validate(logisticRegression.validation_loader,"Validation")
    
    results = dict(
        model=logisticRegression,
    )

    return results

class FNN(nn.Module):
    # source : https://drive.google.com/file/d/1VNO4BzvtEhNV62NNnHYcD_ZBOdf9dxAB/view?usp=sharing
    def __init__(self, loss_type, num_classes):
        super(FNN, self).__init__()

        self.loss_type = loss_type
        self.num_classes = num_classes

        self.fc1 = nn.Linear(32*32*3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        output = x.view(x.size(0), -1)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        output = F.softmax(output)
        return output

    def get_loss(self, output, target):
        if self.loss_type == "ce":
            loss = F.cross_entropy(output, target)
        else:
            loss = F.mse_loss(output, target)
        return loss


def tune_hyper_parameter(target_metric, device):
    # TODO: implement logistic regression and FNN hyper-parameter tuning here
    #metric = accuracy
    # source: https://drive.google.com/file/d/1GTR1thHyW7TDtC-yWUYHZf6QyjzQREVd/view?usp=sharing
    best_params = []
    best_metric = []
    
    # Part 1
    model = logistic_regression(device)['model']
    best_accuracy = 0.0
    best_combo = {}
    for i in torch.arange(1, -6, -1):
        lr = np.power(10.0, i)
        for j in torch.arange(1, -6, -1):
            wd = np.power(10.0, j)
            optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=wd)
            model.start_training(model.validation_loader, optimizer)
            accuracy = model.validate(model.validation_loader, "Validation")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_combo = {"Logistic Regression": {"learning_rate": lr, "weight_decay": wd}}

    best_params.append(best_combo)
    best_metric.append({f"Logistic Regression {target_metric}": best_accuracy})
    
    #Part 2
    
    def get_dataloaders(batch_size):
        """

        :param Params.BatchSize batch_size:
        :return:
        """

        CIFAR_training = datasets.CIFAR10('.', train=True, download=True,
                                                    transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        # create a training and a validation set
        CIFAR_train_set, CIFAR_val_set = random_split(CIFAR_training, [40000, 10000])

        train_loader = DataLoader(CIFAR_train_set, batch_size=batch_size, shuffle=True)

        val_loader = DataLoader(CIFAR_val_set, batch_size=batch_size, shuffle= False)


        return train_loader, val_loader

    def trainModel(net, optimizer, train_loader, device):
        net.train()
        pbar = tqdm(train_loader, ncols=100, position=0, leave=True)
        avg_loss = 0
        for batch_idx, (data, target) in enumerate(pbar):
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            loss = net.get_loss(output, target)
            loss.backward()
            optimizer.step()

            loss_sc = loss.item()

            avg_loss += (loss_sc - avg_loss) / (batch_idx + 1)

            pbar.set_description('train loss: {:.6f} avg loss: {:.6f}'.format(loss_sc, avg_loss))


    def validation(net, validation_loader, device):
        net.eval()
        validation_loss = 0
        correct = 0
        for data, target in validation_loader:
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            loss = net.get_loss(output, target)
            validation_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

        validation_loss /= len(validation_loader.dataset)
        print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            validation_loss, correct, len(validation_loader.dataset),
            100. * correct / len(validation_loader.dataset)))
        return 100. * correct / len(validation_loader.dataset)

    train_loader, val_loader = get_dataloaders(5000)
    model = FNN("ce", 10)
    best_accuracy = 0.0
    best_combo = {}
    for i in torch.arange(1, -6, -1):
        lr = np.power(10.0, i)
        for j in torch.arange(1, -6, -1):
            wd = np.power(10.0, j)
            optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=wd)
            trainModel(model, optimizer, train_loader, device)
            accuracy = validation(model, val_loader, device=device)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_combo = {"FNN": {"learning_rate": lr, "weight_decat": wd}}

            
    best_params.append(best_combo)
    best_metric.append({f"FNN {target_metric}": best_accuracy})
    return best_params, best_metric
