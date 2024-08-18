
import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from random import randrange
import torchvision.transforms as transforms


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp", ".JPG", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        dir_path = join(self.data_dir, str(index + 1))
        
        # Check if directory exists
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        # List all image files in the directory
        data_filenames = [join(dir_path, x) for x in listdir(dir_path) if is_image_file(x)]
        num = len(data_filenames)
        
        # Check if there are any images in the directory
        if num == 0:
            raise FileNotFoundError(f"No image files found in directory: {dir_path}")
        
        index1 = random.randint(1, num)
        index2 = random.randint(1, num)
        
        # Ensure index1 and index2 are different
        while index1 == index2:
            index2 = random.randint(1, num)

        im1 = load_img(data_filenames[index1 - 1])
        im2 = load_img(data_filenames[index2 - 1])

        _, file1 = os.path.split(data_filenames[index1 - 1])
        _, file2 = os.path.split(data_filenames[index2 - 1])

        seed = np.random.randint(123456789)
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)
        
        return im1, im2, file1, file2

    
    def __len__(self):
        return len(self.data_filenames)
