# train the model
import torch
import torch.nn as nn
from data import OCRDataset
from torch.utils.data import DataLoader

# train the model over one epochs
def train_epoch(device, model, dataloader, optimizer, criterion):
    # put model into training mode
    model.train()

    # initialize loss
    total_loss = 0

    # run the model on each batch
    for batch_no, (noisy, denoised) in enumerate(dataloader):
        # send inputs to GPU for faster computation
        noisy, denoised = noisy.to(device), denoised.to(device)

        # clear the gradients from the last batch
        optimizer.zero_grad()

        # do the forward pass
        pred_denoised = model(noisy)

        # calculate and accumulate the loss (https://arxiv.org/pdf/1608.03981)
        gt_noise = denoised - noisy
        pred_noise = pred_denoised - noisy
        loss = criterion(gt_noise, pred_noise)
        total_loss += loss 

        # compute all the gradients for back propagation
        loss.backward()

        # update the model weights
        optimizer.step()

    # normalize the total loss by the number of batches we had
    train_loss = total_loss / len(dataloader)

    return train_loss