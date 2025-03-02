import torch.utils.data as data
import os
from utils import download_and_convert
import csv
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torchvision.transforms as transforms

class PetDataset(data.Dataset):
    str2label = {"cat": 0, "dog": 1}

    def __init__(self, train, **kwargs):
        super(PetDataset, self).__init__()

        self.size = kwargs.get("size", None)
        self.data_root = kwargs.get("data_root", "./data")

        # download and convert the dataset
        self._prepare_dataset(self.data_root)

        # read data
        images = list()
        labels = list()
        csv_name = "train.csv" if train else "test.csv"
        print(os.path.join(self.data_root, "pet", csv_name))
        with open(os.path.join(self.data_root, "pet", csv_name)) as f:
            reader = csv.reader(f)
            nr_lines = 0
            for line in reader:
                nr_lines +=1
                (filename, label) = line
                # read image
                image = Image.open(filename).convert('RGB')
                # resize input images
                if self.size:
                    image = image.resize((self.size, self.size), Image.BICUBIC)
                # create data list
                images.append(image)
                labels.append(label)

        # convert lists to numpy array
        images = np.array(images, dtype="float32")
        labels = np.array(labels)

        # label encoding
        le = LabelEncoder()
        labels = le.fit_transform(labels)   

        # convert numpy arrays to PyTorch tensors
        self.images = torch.tensor(images)
        self.labels = torch.tensor(labels)

        # define a transform to apply to the images
        self.transform = transforms.Compose([
            transforms.ToPILImage(), # pixel to range [0,1]
            transforms.ToTensor(),
            transforms.Normalize(mean=kwargs.get("mean", None), std=kwargs.get("std", None))
        ])

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # transpose from HxWxC to CxHxW
        image = image.permute(2, 0, 1)

        # normalize image and converto to tensor
        if self.transform:
            image = self.transform(image)

        return (image, label)

    def _prepare_dataset(self, data_root):
        check = os.path.join(data_root, "pets")
        if not os.path.isdir(check):
            download_and_convert(data_root)

    def __len__(self):
        return self.images.size(0)

        