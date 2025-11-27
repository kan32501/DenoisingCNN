# this document pre-processes data for analysis

# image processing
import os
import matplotlib.pyplot as plt
import PIL.Image as Image

# PyTorch 
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

# process the dataset to be a standardized set of images
class OCRDataset(Dataset):
    def __init__(self, noisy_path, denoised_path, transform=None):
        # initialize base attributes
        self.noisy_path = noisy_path
        self.denoised_path = denoised_path

        # get each item. sort them so the pairs are always aligned
        raw_noisy_images = sorted([os.path.join(self.noisy_path, file) for file in os.listdir(self.noisy_path)])
        raw_denoised_images = sorted([os.path.join(self.denoised_path, file) for file in os.listdir(self.denoised_path)])
    
        # if there is an unequal number of images throw an error
        if len(raw_noisy_images) != len(raw_denoised_images):
            raise valueError("Unequal number of noisy and denoised images")

        # set the standard transformation them to be 128 x 128 tensor
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        # load & standardize each pair of images
        self.noisy_images = []
        self.denoised_images = []
        for i in range(len(raw_noisy_images)):
            # load image
            noisy = Image.open(raw_noisy_images[i])
            denoised = Image.open(raw_denoised_images[i])

            # preprocess image
            self.noisy_images.append(self.transform(noisy))
            self.denoised_images.append(self.transform(denoised))

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        return (self.noisy_images[idx], self.denoised_images[idx])
