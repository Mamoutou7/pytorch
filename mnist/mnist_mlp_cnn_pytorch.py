#!/usr/bin/python
# -*- coding: utf-8 -*-

# Classification de MNIST with a Multi Layer Perceptron(MLP) or a Convolution Neural Network(CNN)

import matplotlib.pyplot as plt
import torch
import argparse
from torch.optim.lr_scheduler import StepLR

# import datasets
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

def data_loader(train_data, test_data, batch_size):

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True

    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False

    )

    #print('total training batch number: {}'.format(len(train_loader)))
    #print('total testing batch number: {}'.format(len(test_loader)))

    return train_loader, test_loader


def imshow(tensor, title=None):
    '''
    Display some images
    :return:
    '''

    img = tensor.cpu().clone()
    img = img.squeeze()

    plt.imshow(img, cmap="gray")
    if title is not None:
        plt.title(title)
    plt.pause(0.5)

plt.figure()
for e in range(10):
    print(train_set.data[e,:,:])
    imshow(train_set.data[e,:,:], title='MNIST example ({})'.format(train_set.targets[e]))
plt.close()


def display_img(train_loader):
    '''
    Display image and label.
    :param train_loader:
    :return:
    '''

    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")



# define MultiLayer Perceptron model

class RegSoftNet(nn.Module):
    def __init__(self, DATA_SIZE, NUM_CLASSES):
        super(RegSoftNet, self).__init__()
        self.DATA_SIZE = DATA_SIZE
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(self.DATA_SIZE, self.NUM_CLASSES)

    def forward(self, x):
        x = x.view(-1, self.DATA_SIZE) # reshape the tensor
        x = self.fc(x)
        return x


class MLP(nn.Module):
    def __init__(self, DATA_SIZE, NUM_CLASSES, NUM_HIDDEN_1, NUM_HIDDEN_2):
        super(MLP, self).__init__()
        self.DATA_SIZE = DATA_SIZE # 784
        self.NUM_CLASSES = NUM_CLASSES # 10
        self.NUM_HIDDEN_1 = NUM_HIDDEN_1 #256  # try 512
        self.NUM_HIDDEN_2 = NUM_HIDDEN_2 #256
        self.fc1 = nn.Linear(self.DATA_SIZE, self.NUM_HIDDEN_1)
        self.fc2 = nn.Linear(self.NUM_HIDDEN_1, self.NUM_HIDDEN_2)
        self.fc3 = nn.Linear(self.NUM_HIDDEN_2, self.NUM_CLASSES)

    def forward(self, x):
        x = x.view(-1, self.DATA_SIZE) # reshape the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class MnistCNN(nn.Module):
    def __init__(self, NUM_CONV_1, NUM_CONV_2, NUM_CLASSES):
        super(MnistCNN, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.NUM_CONV_1 = NUM_CONV_1 #10
        self.NUM_CONV_2 = NUM_CONV_2 #20
        self.NUM_FC = 500
        self.conv_1 = nn.Conv2d(1, self.NUM_CONV_1, 5, 1) # kernel_size = 5
        self.conv_2 = nn.Conv2d(self.NUM_CONV_1, self.NUM_CONV_2, 5, 1) # kernel_size = 5
        self.dropout_1 = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.5)
        self.fc_1 = nn.Linear(4*4*self.NUM_CONV_2, self.NUM_FC)
        self.fc_2 = nn.Linear(self.NUM_FC, self.NUM_CLASSES)

    def forward(self, x):

        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*self.NUM_CONV_2)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x
        # en utilisant loss = F.nll_loss(output, target) on peut faire
        # return F.log_softmax(x, dim=1)


def model_train(args, model, device, train_loader,  optimizer, epoch, loss_fn):
    '''
    Model train function
    :param args:
    :param model:
    :param device:
    :param train_loader:
    :param optimizer:
    :param epoch:
    :return:
    '''

    for epoch in range(10):
        # training
        model.train() # mode "train" agit sur "dropout" ou "batchnorm"
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = x.to(device), target.to(device)
            out = model(x)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('epoch {} batch {} [{}/{}] training loss: {}'.format(epoch, batch_idx, batch_idx*len(x),
                        len(train_loader.dataset),loss.item()))

def model_test(model, device, test_loader, loss_fn):
    '''
    Model function test
    :param model:
    :param device:
    :param test_loader:
    :return:
    '''
    # testing
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_loader):
            x, target = x.to(device), target.to(device)
            out = model(x)
            loss = loss_fn(out, target)
            # _, prediction = torch.max(out.data, 1)
            prediction = out.argmax(dim=1, keepdim=True) # index of the max log-probability
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    taux_classif = 100. * correct / len(test_loader.dataset)
    print('Accuracy: {}/{} (tx {:.2f}%, err {:.2f}%)\n'.format(correct, len(test_loader.dataset), taux_classif, 100.-taux_classif))


def save_model(model, is_cnn=False):
    # BONUS: save model to disk (for further inference)
    if is_cnn == True:
        filename = 'model_cnn.pth'
        torch.save(model.state_dict(), "params_"+filename)
        torch.save(model, filename)
        print("saved model to {}".format(filename))
    else:
        filename = 'model_mlp.pth'
        torch.save(model.state_dict(), "params_"+filename)
        torch.save(model, filename)
        print("saved model to {}".format(filename))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')


    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_set = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_set = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()

    )

    train_loader, test_loader = data_loader(train_set, test_set, args.batch_size)

    # optimization hyperparameters
    model = MnistCNN.to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr)  # try lr=0.05, lr=0.01, momentum=0.9

    loss_fn = nn.CrossEntropyLoss()

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        model_train(args, model, device, train_loader, optimizer, epoch)
        model_test(model, device, test_loader)

        scheduler.step()

    if args.save_model:
        save_model(model)


if __name__ == '__main__':
    main()