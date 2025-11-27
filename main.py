# test that the dataset can be loaded
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse

# data 
noisy_path = "./data/noisy_images"
denoised_path = "./data/denoised_images"

# load the dataset
from data import OCRDataset
dataset = OCRDataset(noisy_path, denoised_path)
len_dataset = len(dataset)

# get a random pair
import random
random_idx = random.randint(0, len_dataset)
noisy, denoised = dataset[random_idx]

# convert back to PIL
noisy_pil = transforms.ToPILImage()(noisy)
denoised_pil = transforms.ToPILImage()(denoised)

# show images
fig, axs = plt.subplots(1, 2)
axs[0].imshow(noisy_pil, cmap='gray')
axs[1].imshow(denoised_pil, cmap='gray')
plt.savefig('./random_pair.png')

# make sure the model doesn't throw any errors:
from model import DenoisingCNN
model = DenoisingCNN()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--noisy_path', type=str, default="./data/noisy_images")
    parser.add_argument('--denoised_path', type=str, default="./data/denoised_images")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=5)

# specify 
def main():
    # get the arguments
    args = parse_args()

    # initialize the GPU
    device = torch.device("cuda" if torch.cuda_is_available() else "cpu")

    # load all the data, split, and load into a DataLoader
    dataset = OCRDataset(args.noisy_path, args.denoised_path)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1]) # we don't have that much data
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) # shuffle ordering after each epoch
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) # don't need to shuffle for test

    # define the loss function which is the MSE between the noise from the GT pair and the predicted pair
    criterion = nn.MSELoss()

    # train the model
    

    # test/evaluate the model
