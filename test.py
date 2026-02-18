# test the model
import torch
import torch.nn as nn
from data import OCRDataset
from torch.utils.data import DataLoader

# test the model
def test_epoch(device, model, dataloader, criterion):
    # set model to evaluation mode
    model.eval()

    # initialize testing loss
    total_loss = 0

    # only inference, so don't compute gradients
    with torch.no_grad():
        # compare prediction & target for each data pair
        for batch_no, (noisy, denoised) in enumerate(dataloader):
            # send tensors to GPU
            noisy, denoised = noisy.to(device), denoised.to(device)
            batch_size = noisy.size()[0]

            # get model prediction
            pred = model(noisy)

            # calculate and accumulate the loss (https://arxiv.org/pdf/1608.03981)
            gt_noise = noisy - denoised
            loss = criterion(gt_noise, pred)
            # loss = criterion(denoised, pred) # if we are directly predicting the denoised image
            total_loss += loss 

    # normalize the loss with the number of batches
    test_loss = total_loss / len(dataloader)

    return test_loss