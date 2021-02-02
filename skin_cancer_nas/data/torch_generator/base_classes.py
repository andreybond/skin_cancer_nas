import os
import logging
import torch
import cv2
import numpy as np
import random
from random import randrange

import traceback

from config import VALID_CHANNELS, MISSING_CH_IMG, IMG_HEIGHT, IMG_WIDTH

logger = logging.getLogger('nni')
APPLY_RANDOM_INTERCHANNEL_ROTATION = False

random_rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
channels_to_drop = [['r-r'], ['ir-r'], ['uv-0-r', 'uv-0-g']] # these channels can be zeroed out (controlled by probability variable - channel_drop_prob)

def apply_channels_rnd_rotation(img):
    choise = random.choice(random_rotations)
    if choise is not None:
        return cv2.rotate(img, choise)
    else:
        return img

import cv2

def pad_img_to_square(desired_size = 320, img=None):

    assert img is not None

    im = img  #cv2.imread(im_pth)
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_REPLICATE, value=color)
    return new_im

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sample_paths, labels, transform=None, device=None, debug=False, valid_channels=VALID_CHANNELS, channels_to_zero_out=[], channel_drop_prob=0.0):
        
        assert device is not None

        # logger.info('valid_channels = [{}]'.format(valid_channels))

        self.labels = labels
        sample_paths.sort()
        self.sample_paths = sample_paths
        self.transform = transform
        self.device = device
        self.debug = debug
        self.valid_channels = valid_channels
        self.channels_to_zero_out = channels_to_zero_out
        self.channel_drop_prob = channel_drop_prob  # if not 0.0, then one of the channels from drop_p can be zeroed out (used as regularization)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.sample_paths)

    def __getitem__(self, index):
        # logger.info('__getitem__({})'.format(index))
        '''
        Generates one sample of data
        '''
        try:
            sample_path = self.sample_paths[index]
            if self.debug:
                logger.info('    sample_path={}, index={}'.format(sample_path, index))
            x_out = self._convert_img_to_array(sample_path, self.valid_channels, self.channels_to_zero_out, self.channel_drop_prob)
            x_out = x_out.astype('float32')
            x_out /= 255

            X = torch.as_tensor(x_out)
            y = self.labels[str(sample_path)]
            if self.debug:
                logger.info('           y={}'.format(y))

            if self.transform is not None:
                X = self.transform(X)
            
            X = X.to(self.device)
            y = torch.tensor(y).to(self.device)
            return X, y
        except Exception as e:
            tb = traceback.format_exc()
            logger.error('Exception! traceback={}'.format(tb))
            logger.error('Exception at index={}, e={}'.format(index, e))

    @staticmethod
    def _convert_img_to_array(sample_path, valid_channels, channels_to_zero_out, channel_drop_prob):
        #logger.info('_convert_img_to_array={}'.format(sample_path))
        'Converts n grayscale images to 3D array with n channels'
        x_array = []
        drop_one_channel = random.uniform(0, 1) < channel_drop_prob
        if drop_one_channel:
            channel_to_drop = channels_to_drop[randrange(len(channels_to_drop))]
        else:
            channel_to_drop = []
        for channel in valid_channels:
            channel_path = os.path.join(sample_path, channel)
            if not os.path.exists(channel_path) or channel in channels_to_zero_out or channel in channel_to_drop:
                x_array.append(MISSING_CH_IMG)
                continue
            
            image = os.listdir(channel_path)
            if not image:
                x_array.append(MISSING_CH_IMG)
            
            img = cv2.imread(os.path.join(channel_path, image[0]), flags=cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                x_array.append(MISSING_CH_IMG)
                continue
            else:
                shape = img.shape
                # logger.info(shape)
                img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
                # img = pad_img_to_square(640, img)
                # img = pad_img_to_square(320, cv2.resize(img, (320, 240), interpolation=cv2.INTER_CUBIC))
            
            if APPLY_RANDOM_INTERCHANNEL_ROTATION:
                # now for one of the experiments we want to Randomly (!) rotate channels to misalign them.
                img = apply_channels_rnd_rotation(img)

            x_array.append(img)
        # logger.info('_convert_img_to_array, len(x_array)={}'.format(len(x_array)))
        return np.stack(x_array, axis=0)