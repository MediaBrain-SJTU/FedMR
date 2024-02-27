import torch.nn as nn
import torch.nn.functional as F
from resnetcifar import ResNet18_cifar10, ResNet50_cifar10,ResNet18_mnist
from resnet import *
import torch
import torchvision.models as models
from collections import OrderedDict
import torchvision.transforms as transforms

class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        return x

class SimpleFemnist(nn.Module):
    def __init__(self):
        super(SimpleFemnist, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(1, 32, 5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))

        self.fc_layer=nn.Sequential(
            # nn.Linear(4096, 1024),
            nn.Linear(7 * 7 * 64, 2048),
            nn.ReLU(inplace=True),

        )
        self.cls=nn.Linear(2048, 62)

    def forward(self, x):
        h = self.conv_layer(x)
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        h = h.view(-1, 7 * 7 * 64)
        x = self.fc_layer(h)
        y = self.cls(x)
        return h, x, y

class model_synthetic(nn.Module):
    def __init__(self):
        super(model_synthetic, self).__init__()
        self.len = 0
        self.loss = 0
        self.fc1 = nn.Linear(60, 20)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.fc4 = nn.Linear(20, 5)

    def forward(self, data):
        h = self.fc1(data)
        h = self.sigmoid(h)
        x = self.fc4(h)
        x = self.sigmoid(x)
        return h,h,x

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=10):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        feature=out
        out = self.linear(out)

        return 0,feature,out


class SimpleCNNMNIST(nn.Module):
    def __init__(self, args , n_classes,out_dim=256):
        super(SimpleCNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.cls = nn.Linear(16*4*4, n_classes)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        y=self.cls(x)
        return x, x, y
class SimpleModel(nn.Module):
    def __init__(self, n_classes):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.cls = nn.Linear(400, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        y=self.cls(x)
        return y
class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output
class C1_SVHN(nn.Module):
    def __init__(self):
        super(C1_SVHN, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output

class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()

        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c2(img)
        return output


class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()

        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.c3(img)
        return output


class F4(nn.Module):
    def __init__(self):
        super(F4, self).__init__()

        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120, 84)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f4(img)
        return output


class F5(nn.Module):
    def __init__(self):
        super(F5, self).__init__()

        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.f5(img)
        return output

class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.pad = transforms.Pad(2)
        self.c1 = C1()
        self.c2_1 = C2()
        self.c2_2 = C2()
        self.c3 = C3()
        self.f4 = F4()
        self.f5 = F5()

    def forward(self, img):
        img = self.pad(img)
        x = self.c1(img)

        x1 = self.c2_1(x)
        x = self.c2_2(x)

        x += x1


        x = self.c3(x)
        x = x.view(img.size(0), -1)
        x = self.f4(x)
        output = self.f5(x)
        return x,x,output
class VGG16(nn.Module):
    def __init__(self,args):
        super(VGG16, self).__init__()
        if args.dataset=="fmnist":
            input_channel=1
        elif args.dataset=="SVHN":
            input_channel=3
        self.block1=nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=(3, 3),padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3),padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3),padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.block4=nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3),padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        #self.l1=nn.Linear(512, 4096)
        self.cls=nn.Linear(512, 10)
    def forward(self, img):

        h = self.block1(img)
        h=self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = h.view(img.size(0), -1)
        #print(h.shape)
        x=self.cls(h)
        #out=self.l2(x)


        return h,h,x
class LeNet5_SVHN(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet5_SVHN, self).__init__()
        self.c1 = C1_SVHN()
        self.c2_1 = C2()
        self.c2_2 = C2()
        self.c3 = C3()
        self.f4 = F4()
        self.f5 = F5()

    def forward(self, img):

        x = self.c1(img)

        x1 = self.c2_1(x)
        x = self.c2_2(x)

        x += x1


        x = self.c3(x)
        x = x.view(img.size(0), -1)
        x = self.f4(x)
        output = self.f5(x)
        return x,x,output
#class simplemodel(nn.Module):

class model_cifar(nn.Module):
    def __init__(self, args , n_classes,out_dim=256):
        super(model_cifar, self).__init__()

        if args.model == "resnet50":
            basemodel = ResNet50_cifar10()
            self.backbone = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif args.model=="resnet18":
            if args.dataset in ["mnist","fmnist"]:
                basemodel=ResNet18_mnist()
            else:
                basemodel = ResNet18_cifar10()
            self.backbone = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif args.model == 'simple-cnn':
            self.backbone = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84])
            num_ftrs = 84
        self.cls=nn.Linear(num_ftrs,n_classes)

    def forward(self, x):
        h = self.backbone(x)
        h=h.squeeze()
        x=h

        y = self.cls(x)
        return h, x, y










