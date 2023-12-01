#
# Implementation of AlexNet for illustrative purposes. The train.py driver
# can import AlexNet from here or directly from torchvision.
#
# Taken from torchvision.models.alexnet:
# https://pytorch.org/docs/1.6.0/_modules/torchvision/models/alexnet.html#alexnet


import torch
import torch.nn as nn
from pydebug import gd, infoTensor

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        gd.debuginfo(prj="ds_chat", info=f'num_classes={num_classes}')
        # https://blog.csdn.net/dongjinkun/article/details/114575998
        # Python中的super(Net, self).__init__()是指首先找到Net的父类（比如是类NNet），
        # 然后把类Net的对象self转换为类NNet的对象，然后“被转换”的类NNet对象调用自己的init函数，
        # 其实简单理解就是子类把父类的__init__()放到自己的__init__()当中，
        # 这样子类就有了父类的__init__()的那些东西
        super(AlexNet, self).__init__()
        gd.debuginfo(prj="ds_chat")

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        gd.debuginfo(prj="ds_chat", info=f'self.features={self.features}')
        gd.debuginfo(prj="ds_chat", info=f'self.avgpool={self.avgpool}')
        gd.debuginfo(prj="ds_chat", info=f'self.classifier={self.classifier}')

    def forward(self, x):
        gd.debuginfo(prj="ds_chat", info=f'===============================')
        x = self.features(x)
        gd.debuginfo(prj="ds_chat", info=f'x={x}')

        x = self.avgpool(x)
        gd.debuginfo(prj="ds_chat", info=f'x={x}')

        x = torch.flatten(x, 1)
        gd.debuginfo(prj="ds_chat", info=f'x={x}')

        x = self.classifier(x)
        gd.debuginfo(prj="ds_chat", info=f'x={x}')

        return x
