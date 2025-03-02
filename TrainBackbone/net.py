from torch.nn import Module
from torchvision.models import resnet50
from torchvision.models import vgg19
from torchvision.models import ResNet50_Weights
from torchvision.models import VGG19_Weights
from torch.nn import Identity
from torch.nn import Sequential
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Dropout
from torch.nn import MaxPool2d
from torch.nn import Conv2d
import torch

class Net(Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.conv_layers = Sequential(
            Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)
        )

        n_inputs = 128*7*7

        # build the classifier for class prediction
        self.classifier = Sequential(
            Linear(n_inputs, 512), 
            ReLU(),
            Dropout(p=0.5),
            Linear(512, 256),
            ReLU(),
            Dropout(p=0.5),
            Linear(256, num_classes)
        )

    def forward(self, x):
        conv1 = self.conv_layers(x)
        flat = conv1.view(conv1.size(0), -1)
        classLogits = self.classifier(flat)
        return classLogits