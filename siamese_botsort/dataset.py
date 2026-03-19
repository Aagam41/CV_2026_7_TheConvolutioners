# Usage: import only — not run directly. Used by train.py via SiameseDataset.
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

class SiameseDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.obj_folders = []
        self.same_folders = []

        for folder in os.listdir(root):
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                continue

            imgs = [
                name for name in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, name))
            ]

            if len(imgs) >= 1:
                self.obj_folders.append(folder)
            if len(imgs) >= 2:
                self.same_folders.append(folder)

        if len(self.obj_folders) < 2:
            raise ValueError(f"Need at least 2 object folders in {root}.")
        if len(self.same_folders) == 0:
            raise ValueError(f"Need at least 1 folder with 2+ images in {root}.")

    def __len__(self):
        return 10000  # arbitrary

    def __getitem__(self, idx):
        same = random.choice([0, 1])

        if same:
            folder = random.choice(self.same_folders)
            imgs = os.listdir(os.path.join(self.root, folder))
            img1, img2 = random.sample(imgs, 2)
            label = 1
        else:
            f1, f2 = random.sample(self.obj_folders, 2)
            img1 = random.choice(os.listdir(os.path.join(self.root, f1)))
            img2 = random.choice(os.listdir(os.path.join(self.root, f2)))
            folder = None
            label = 0

        if same:
            path1 = os.path.join(self.root, folder, img1)
            path2 = os.path.join(self.root, folder, img2)
        else:
            path1 = os.path.join(self.root, f1, img1)
            path2 = os.path.join(self.root, f2, img2)

        img1 = transform(Image.open(path1).convert("RGB"))
        img2 = transform(Image.open(path2).convert("RGB"))

        return img1, img2, label