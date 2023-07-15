import torch
from torch.utils.data import Dataset
import torch.nn.functional as func
import torchvision.models as models
import torchvision.transforms as tfms
from PIL import Image
from skimage.draw import random_shapes

# 不知道哪里找到的野鸡模板，后面改

class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.transform = transform
        self.images = []
        self.labels = []
        self.load_data()

    def load_data(self):
        for i in range(1, 101):
            img = Image.open(self.root_dir + str(i) + '.png')
            self.images.append(img)
            self.labels.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class GNNDataset(Dataset):
    def __init__(self, root_dir):
        pass