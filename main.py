# test that the dataset can be loaded
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import test
import train

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

# write a function that plots the noisy, denoised, and predicted images of this model
# w/ plt
def visualize(model, noisy, denoised):
    pass

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--noisy_path', type=str, default="./data/noisy_images")
    parser.add_argument('--denoised_path', type=str, default="./data/denoised_images")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--train_test_split', type=float, default=0.9) # we don't have that much data
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--visualize', type=bool, default=True)

    args = parser.parse_args()

    return args

# specify 
def main():
    # get the arguments
    args = parse_args()

    # initialize the GPU
    device = torch.device("cuda" if torch.cuda_is_available() else "cpu")

    # load all the data, split, and load into a DataLoader
    dataset = OCRDataset(args.noisy_path, args.denoised_path)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [args.train_test_split, 1.0-args.train_test_split]) 
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) # shuffle ordering after each epoch
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) # don't need to shuffle for test

    # initialize the model (https://arxiv.org/pdf/1608.03981) and send to GPU
    model = DenoisingCNN(num_layers=args.num_layers).to(device)

    # define the loss function (MSE between the noise from the GT pair and the predicted pair)
    criterion = nn.MSELoss()

    # choose the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

    # train the model for N epochs, test after each
    epoch_losses = []
    for epoch in tqdm(range(1, epochs + 1)):
        # train the model
        train_loss = train_epoch(device, model, train_dataloader, optimizer, criterion)

        # test the model
        test_loss = test_epoch(device, model, test_dataloader, criterion)

        # bookkeep losses
        epoch_losses.append((train_loss, test_loss))

        # print losses
        print(
            f"\nEPOCH {epoch} | "
            f"TRAINING LOSS: {train_loss:.5f} / "
            f"TESTING LOSS: {test_loss:.5f} "
        )

    # visualize 5 or so predictions


    # START HERE: RUN THE CODE