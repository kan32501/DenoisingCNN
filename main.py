# test that the dataset can be loaded
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import random

from data import OCRDataset, random_pairs, plot_images
from model import DenoisingCNN
from train import train_epoch
from test import test_epoch

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--noisy_path', type=str, default="./data/noisy_images")
    parser.add_argument('--denoised_path', type=str, default="./data/denoised_images")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--train_test_split', type=float, default=0.9) # we don't have that much data
    parser.add_argument('--num_layers', type=int, default=17)
    parser.add_argument('--visualize', type=bool, default=True)

    args = parser.parse_args()

    return args

# write a function that plots the noisy, denoised, and predicted images of this model
# w/ plt
def visualize_prediction(device, model, test_dataset, reconstruct_fn, samples=3, out_path="./model_preds.png"):
    # get random pairs
    random_idxs = random.sample(range(len(test_dataset)), samples)
    random_test_pairs = [test_dataset[idx] for idx in random_idxs]

    # initialize plot
    fig, axs = plt.subplots(samples, 3, figsize=(3 * 3, samples * 3))

    # run the prediction and visualize
    for i in range(len(random_test_pairs)):
        noisy = random_test_pairs[i][0]
        denoised = random_test_pairs[i][1]
        prediction = model(noisy.unsqueeze(0).to(device)).squeeze(0).cpu().detach()

        # reconstruct each image
        noisy_PIL = reconstruct_fn(noisy)
        denoised_PIL = reconstruct_fn(denoised)
        prediction_PIL = reconstruct_fn(prediction)

        # plot
        axs[i, 0].imshow(noisy_PIL, interpolation='nearest', cmap='gray', aspect='equal')
        axs[i, 1].imshow(denoised_PIL, interpolation='nearest', cmap='gray', aspect='equal')
        axs[i, 2].imshow(prediction_PIL, interpolation='nearest', cmap='gray', aspect='equal')

    for ax in axs.flat:
        ax.axis("off")  # hide ticks for cleaner images

    plt.savefig(out_path)

# plot the graph with losses
def visualize_loss(losses, out_path="./losses.png"):
    # create plot
    fig, axs = plt.subplots(1, 2)

    # plot both training and test losses
    x = [i for i in range(1, len(losses) + 1)]
    train_losses = [loss[0] for loss in losses]
    test_losses = [loss[1] for loss in losses] # train or test loss
    axs[0].plot(x, train_losses)
    axs[1].plot(x, test_losses)

    plt.savefig(out_path)



# specify 
def main():
    # get the arguments
    args = parse_args()

    # initialize the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load all the data, split, and load into a DataLoader
    dataset = OCRDataset(args.noisy_path, args.denoised_path)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [args.train_test_split, 1.0-args.train_test_split]) 
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) # shuffle ordering after each epoch
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) # don't need to shuffle for test

    # show data
    random_noised, random_denoised = random_pairs(dataset)
    plot_images(random_noised, random_denoised, out_path="./random_pair.png")

    # initialize the model (https://arxiv.org/pdf/1608.03981) and send to GPU
    model = DenoisingCNN(num_layers=args.num_layers).to(device)

    # define the loss function (MSE between the noise from the GT pair and the predicted pair)
    criterion = nn.MSELoss()

    # choose the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

    # train the model for N epochs, test after each
    epoch_losses = []
    for epoch in tqdm(range(1, args.epochs + 1)):
        # train the model
        train_loss = train_epoch(device, model, train_dataloader, optimizer, criterion)

        # test the model
        test_loss = test_epoch(device, model, test_dataloader, criterion)

        # bookkeep losses
        epoch_losses.append((train_loss.cpu().detach(), test_loss.cpu().detach()))

        # print losses
        print(
            f"\nEPOCH {epoch} | "
            f"TRAINING LOSS: {train_loss:.5f} / "
            f"TESTING LOSS: {test_loss:.5f} "
        )

        # visualize losses
        visualize_loss(epoch_losses)

    # visualize 3 predictions with trained model
    visualize_prediction(device, model, test_dataset, dataset.reconstruct_image)

if __name__ == "__main__":
    main()