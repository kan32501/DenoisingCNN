# this document pre-processes data for analysis

# image processing
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import random
from tqdm import tqdm

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
        self.image_size = 128

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
        for i in tqdm(range(len(raw_noisy_images)), desc="Loading dataset"):
            # load image
            noisy = Image.open(raw_noisy_images[i])
            denoised = Image.open(raw_denoised_images[i])

            # # set original image size
            # if i == 0: self.original_size = noisy.size

            # preprocess image
            self.noisy_images.append(self.transform(noisy))
            self.denoised_images.append(self.transform(denoised))

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        return (self.noisy_images[idx], self.denoised_images[idx])

    # assumes 128x128 tensor input
    def reconstruct_image(self, tensor):
        # resize back to the original size
        self.un_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToPILImage()
        ])

        # bring it off the GPU
        reconstructed = self.un_transform(tensor).cpu().detach() if tensor.is_cuda else self.un_transform(tensor)

        return reconstructed

# get 5 random pairs from the dataset
def random_pairs(dataset, num_pairs=5):
    # get random indices
    len_dataset = len(dataset)
    random_idxs = random.sample(range(len_dataset), num_pairs)
    random_pairs = [dataset[idx] for idx in random_idxs]

    # convert back to PIL
    noisys_pil = [dataset.reconstruct_image(pair[0]) for pair in random_pairs]
    denoiseds_pil = [dataset.reconstruct_image(pair[1]) for pair in random_pairs]

    return noisys_pil, denoiseds_pil

# plot images
def plot_images(noisy_images, denoised_images, out_path):
    # get number of pairs
    num_pairs = len(noisy_images)

    # plot pair
    fig, axs = plt.subplots(num_pairs, 2, figsize=(2 * 3, num_pairs * 3))
    for pair in range(num_pairs):
        axs[pair, 0].imshow(noisy_images[pair], interpolation='nearest', cmap='gray', aspect='equal')
        axs[pair, 1].imshow(denoised_images[pair], interpolation='nearest', cmap='gray', aspect='equal')

    for ax in axs.flat:
        ax.axis("off")  # hide ticks for cleaner images

    # save
    plt.savefig(out_path)
