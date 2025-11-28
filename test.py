# test the model
import torch
import torch.nn as nn
from data import OCRDataset
from torch import DataLoader

# test the model
def test_epoch(device, model, dataloader, criterion, visualize=False):
    # set model to evaluation mode
    model.eval()

    # initialize testing loss
    total_loss = 0

    # only inference, so don't compute gradients
    with torch.no_grad():
        # compare prediction & target for each data pair
        for batch_no, (noisy, denoised) in enumerate(dataloader):
            # send tensors to GPU
            noisy, denoised = noisy.to(device), denoise.to(device)

            # get model prediction
            pred_denoised = model(noisy)

            # calculate and accumulate the loss (https://arxiv.org/pdf/1608.03981)
            gt_noise = denoised - noisy
            pred_noise = pred_denoised - noisy
            loss = criterion(gt_noise, pred_noise)
            total_loss += loss 

    # normalize the loss with the number of batches
    total_loss /= len(dataloader)