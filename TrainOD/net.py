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
from custom_backbone import CustomBackbone

import torch

class Net(Module):
    def __init__(self, backbone, num_classes):
        super(Net, self).__init__()

        # build the backbone for feature extraction
        if backbone=="resnet":
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            n_inputs = self.backbone.fc.in_features
            self.backbone.fc = Identity()
        elif backbone == "vgg19":
            self.backbone = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
            n_inputs = self.backbone.classifier[0].in_features
            self.backbone.classifier = Identity()
        elif backbone == "custom":
            self.backbone = CustomBackbone(num_classes=num_classes)
            self.backbone.load_state_dict(torch.load(f'custom_backbone_weights/pet_50.pth'))
            n_inputs = self.backbone.classifier[0].in_features
            self.backbone.classifier = Identity()

        # freeze backbone parameter
        for param in self.backbone.parameters():
            param.requires_grad = False

        # build the regressor for bounding box prediction
        self.regressor = Sequential(
			Linear(n_inputs, 128),
			ReLU(),
			Linear(128, 64),
			ReLU(),
			Linear(64, 32),
			ReLU(),
			Linear(32, 4),
			Sigmoid()
		)
        # build the classifier for class prediction
        self.classifier = Sequential(
            Linear(n_inputs, 512), 
            ReLU(),
            Dropout(p=0.6),
            Linear(512, 256),
            ReLU(),
            Dropout(p=0.5),
            Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        bbox = self.regressor(features)
        classLogits = self.classifier(features)
        return (bbox, classLogits)
    