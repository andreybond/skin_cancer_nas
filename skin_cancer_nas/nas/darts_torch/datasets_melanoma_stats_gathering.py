"""
This script utilizes defined PyTorch data loader to caclulate statistics required for images normalization.
It calculates std and mean for each separate channel. 
These values are then hardcoded in the data transformer normalization step.
"""
import sys
import os

sys.path.append('/mnt')
sys.path.append('/mnt/skin_cancer_nas')
sys.path.append('/mnt/skin_cancer_nas/data/torch_generator')

from skin_cancer_nas.data.torch_generator import generator as data_gen  
from skin_cancer_nas.data.torch_generator import base_classes
from skin_cancer_nas.data.torch_generator import config
from skin_cancer_nas.data.torch_generator import preprocessor

import torch
from torchvision import transforms

from base_classes import Dataset
import logging
logger = logging.getLogger('nni')

if __name__ == '__main__':
    logger.info('os.getcwd() = ' + os.getcwd())

    logger.info('Loading data...')
    partition, labels = data_gen.train_val_split(val_size=0.1)

    MEAN = [0.2336, 0.6011, 0.3576, 0.4543]
    STD = [0.0530, 0.0998, 0.0965, 0.1170]
    normalize = [
        transforms.Normalize(MEAN, STD)
    ]
    train_transform = transforms.Compose(normalize)
    valid_transform = transforms.Compose(normalize)

    # Generators Declaration
    training_set = Dataset(partition['train'], labels, transform=train_transform)
    training_generator = torch.utils.data.DataLoader(training_set, **data_gen.PARAMS)

    # validation_set = Dataset(partition['validation'], labels, transform=valid_transform)
    # validation_generator = torch.utils.data.DataLoader(validation_set, **data_gen.PARAMS)

    nimages = 0
    mean = 0.0
    var = 0.0
    for i_batch, batch_target in enumerate(training_generator):
        batch = batch_target[0]
        # logger.info('1. batch.shape = {}'.format(batch.shape))
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)

        # logger.info('2. batch.shape = {}'.format(batch.shape))
        # logger.info('3. batch mean = {}'.format(batch.mean(2).sum(0)))
        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        var += batch.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    logger.info(mean)
    logger.info(std)
