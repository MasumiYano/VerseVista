import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms


class PoemDataLoader(Dataset):
    def __init__(self, poem_dir, transform=None):
        super(PoemDataLoader, self).__init__()
        self.poem_dir = poem_dir
        self.transform = transform
        self.poem_index = []

        for root, _, files in os.walk(self.poem_dir):
            for file in files:
                self.poem_index.append(str(os.path.join(self.poem_dir, file)))

    def __len__(self):
        return len(self.poem_index)

    def __getitem__(self, idx):
        poem_path = self.poem_index[idx]
        with open(poem_path, 'r') as f:
            poem = f.read()

        if self.transform:
            poem = self.transform(poem)

        return poem


transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor()
])

cap = datasets.CocoCaptions(
    root='cocodataset/images',
    annFile='cocodataset/annotation',
    transform=transform
)
