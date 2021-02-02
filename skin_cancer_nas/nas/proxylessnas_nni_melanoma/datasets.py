import os
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_split_list(in_dim, child_num):
    in_dim_list = [in_dim // child_num] * child_num
    for _i in range(in_dim % child_num):
        in_dim_list[_i] += 1
    return in_dim_list

class DataProvider:
    VALID_SEED = 42  # random seed for the validation set

    @staticmethod
    def name():
        """ Return name of the dataset """
        raise NotImplementedError

    @property
    def data_shape(self):
        """ Return shape as python list of one data entry """
        raise NotImplementedError

    @property
    def n_classes(self):
        """ Return `int` of num classes """
        raise NotImplementedError

    @property
    def save_path(self):
        """ local path to save the data """
        raise NotImplementedError

    @property
    def data_url(self):
        """ link to download the data """
        raise NotImplementedError

    @staticmethod
    def random_sample_valid_set(train_labels, valid_size, n_classes):
        train_size = len(train_labels)
        assert train_size > valid_size

        g = torch.Generator()
        g.manual_seed(DataProvider.VALID_SEED)  # set random seed before sampling validation set
        rand_indexes = torch.randperm(train_size, generator=g).tolist()

        train_indexes, valid_indexes = [], []
        per_class_remain = get_split_list(valid_size, n_classes)

        for idx in rand_indexes:
            label = train_labels[idx]
            if isinstance(label, float):
                label = int(label)
            elif isinstance(label, np.ndarray):
                label = np.argmax(label)
            else:
                assert isinstance(label, int)
            if per_class_remain[label] > 0:
                valid_indexes.append(idx)
                per_class_remain[label] -= 1
            else:
                train_indexes.append(idx)
        return train_indexes, valid_indexes


class ImagenetDataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=32, resize_scale=0.08, distort_color=None):

        self._save_path = save_path
        train_transforms = self.build_train_transform(distort_color, resize_scale)
        train_dataset = datasets.ImageFolder(self.train_path, train_transforms)

        if valid_size is not None:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(train_dataset))
            else:
                assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size
            train_indexes, valid_indexes = self.random_sample_valid_set(
                [cls for _, cls in train_dataset.samples], valid_size, self.n_classes,
            )
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

            valid_dataset = datasets.ImageFolder(self.train_path, transforms.Compose([
                transforms.Resize(self.resize_value),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                self.normalize,
            ]))

            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
        else:
            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = None

        self.test = torch.utils.data.DataLoader(
            datasets.ImageFolder(self.valid_path, transforms.Compose([
                transforms.Resize(self.resize_value),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                self.normalize,
            ])), batch_size=test_batch_size, shuffle=False, num_workers=n_worker, pin_memory=True,
        )

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'imagenet'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 1000

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/dataset/imagenet'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download ImageNet')

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self._save_path, 'val')

    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def build_train_transform(self, distort_color, resize_scale):
        print('Color jitter: %s' % distort_color)
        if distort_color == 'strong':
            color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif distort_color == 'normal':
            color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None
        if color_transform is None:
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(resize_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(resize_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                color_transform,
                transforms.ToTensor(),
                self.normalize,
            ])
        return train_transforms

    @property
    def resize_value(self):
        return 256

    @property
    def image_size(self):
        return 224


import sys
sys.path.append('/mnt')
sys.path.append('/mnt/skin_cancer_nas')
sys.path.append('/mnt/skin_cancer_nas/data/torch_generator')
from skin_cancer_nas.data.torch_generator import generator as data_gen
from skin_cancer_nas.data.torch_generator import base_classes
from skin_cancer_nas.data.torch_generator.config import *
from base_classes import Dataset

class C43_D22_v1_DataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=64, test_batch_size=256, valid_size=None,
                 n_worker=16, resize_scale=0.08, distort_color=None, device=None, pin_memory=True):

        assert device is not None

        partition, labels = data_gen.train_val_split(val_ratio=0.1, classes_list=CLASSES_SET)

        MEAN = [0.2336, 0.6011, 0.3576, 0.4543]
        STD = [0.0530, 0.0998, 0.0965, 0.1170]
        normalize = [
            # transforms.Normalize(MEAN, STD),
        ]
        train_transform = self.build_train_transform(distort_color, resize_scale)
        valid_transform = transforms.Compose(normalize)
        
        # Generators Declaration
        training_set = Dataset(partition['train'], labels, 
                            transform=train_transform, 
                            device=device)
        training_generator = torch.utils.data.DataLoader(training_set, **data_gen.PARAMS, pin_memory=pin_memory)
        validation_set = Dataset(partition['validation'], labels, 
                            transform=valid_transform, 
                            device=device)
        validation_generator = torch.utils.data.DataLoader(validation_set, **data_gen.PARAMS, pin_memory=pin_memory)

        self.train = training_generator
        self.valid = validation_generator
        self.test = validation_generator

    @staticmethod
    def name():
        return 'c23_d22_v1'

    @property
    def data_shape(self):
        return 4, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 2

    @property
    def save_path(self):
        #if self._save_path is None:
        #    self._save_path = '/dataset/imagenet'
        #return self._save_path
        raise NotImplementedError('save_path is not used by SkinCancerNas project')

    @property
    def data_url(self):
        raise NotImplementedError('data_url is not used by SkinCancerNas project')

    @property
    def train_path(self):
        #return os.path.join(self.save_path, 'train')
        raise NotImplementedError('train_path is not implemented, SkinCancerNas project uses internal train/test subdivision via Random split (fixed with seed number)')

    @property
    def valid_path(self):
        #return os.path.join(self._save_path, 'val')
        raise NotImplementedError('valid_path is not implemented, SkinCancerNas project uses internal train/test subdivision via Random split (fixed with seed number)')

    def build_train_transform(self, distort_color, resize_scale):
        print('Color jitter: %s' % distort_color)
        if distort_color == 'strong':
            color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif distort_color == 'normal':
            color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None
        if color_transform is None:
            train_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(self.image_size, scale=(resize_scale, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-90, 90)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor()
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(self.image_size, scale=(resize_scale, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-90, 90)),
                transforms.RandomVerticalFlip(p=0.5),
                color_transform,
                transforms.ToTensor()
            ])
        return train_transforms

    @property
    def resize_value(self):
        return 224  #256

    @property
    def image_size(self):
        return 224