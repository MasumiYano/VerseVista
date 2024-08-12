import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os
from torchvision.io import read_image
from torchvision import datasets


class PoemDataset:
    def __init__(self, poem_dir, transform=None, target_transform=None):
        self.poem_dir = poem_dir
        self.transform = transform
        self.target_transform = target_transform
        self.poem_files = []

        for root, _, files in os.walk(self.poem_dir):
            for file in files:
                self.poem_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.poem_dir)

    def __getitem__(self, idx):
        poem_path = self.poem_files[idx]

        with open(poem_path, 'r') as file:
            poem = file.read()

        if self.transform:
            poem = self.transform(poem)

        return poem


transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


coco_train_data = datasets.CocoCaptions(
    root='coco_data/train/images',
    annFile='coco_data/train/annotations',
    transform=transform
)

coco_test_data = datasets.CocoCaptions(
    root='coco_data/test/images',
    annFile='coco_data/test/annotations',
    transform=transform
)


