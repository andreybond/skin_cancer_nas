import os
from random import shuffle
from collections import defaultdict
from typing import DefaultDict, Dict, Tuple, List
import random
import torch
import os
import logging
logger = logging.getLogger('nni')

logger.info(torch.__version__) # 1.5.0

from base_classes import Dataset
from config import CLASSES_SET

# logger.info('generator.py, POS_PATH = '+ str(POS_PATH))

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
PARAMS = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 16}
EPOCHS = 100

# calculated statistics for 4 channels
# mean - tensor([0.2336, 0.6011, 0.3576, 0.4543])
# std - tensor([0.0530, 0.0998, 0.0965, 0.1170])


def train_val_split(val_ratio: float = 0.0, 
                    classes_list: List = [], 
                    shuffle_seed: int = 42) -> Tuple[DefaultDict, Dict]:
    """
    Splits dataset into train and validation.
    Generates two dictionaries required for Torch Generator.
    :param val_ratio: share of validation set
    :return: partition['train'] contains a list of training sample paths
             partition['validation'] contains a list of validation sample paths
             labels contains the associated label for each sample of the dataset

    classes_list = list of ClassesDefinition() objects
    """
    assert len(classes_list) > 0

    partition = defaultdict(list)
    labels = {}
    # pos_samples = os.listdir(str(POS_PATH))
    # neg_samples = os.listdir(str(NEG_PATH))
    # shuffle(pos_samples)
    # shuffle(neg_samples)

    for clazz in classes_list:
        samples_count = clazz.get_samples_count()
        train_threshold = int(samples_count * (1 - val_ratio))
        samples_paths = clazz.get_samples_paths()
        random.Random(shuffle_seed).shuffle(samples_paths)
        for i, sample_path in enumerate(samples_paths):
            if i < train_threshold:
                partition['train'].append(sample_path)
            else:
                partition['validation'].append(sample_path)
            labels[str(sample_path)] = clazz.get_int_label()
            # pos_count = pos_count + 1
            # if pos_count >= 100:
            #     break


    # pos_train_threshold = int(len(pos_samples) * (1 - val_size))
    # # pos_count = 0
    # for i, sample in enumerate(pos_samples):
    #     if i < pos_train_threshold:
    #         partition['train'].append(POS_PATH / sample)
    #     else:
    #         partition['validation'].append(POS_PATH / sample)
    #     labels[str(POS_PATH / sample)] = 1
    #     # pos_count = pos_count + 1
    #     # if pos_count >= 100:
    #     #     break

    # neg_train_threshold = int(len(neg_samples) * (1 - val_size))
    # # neg_count = 0
    # for i, sample in enumerate(neg_samples):
    #     if i < neg_train_threshold:
    #         partition['train'].append(NEG_PATH / sample)
    #     else:
    #         partition['validation'].append(NEG_PATH / sample)
    #     labels[str(NEG_PATH / sample)] = 0
    #     # neg_count = neg_count + 1
    #     # if neg_count >= pos_count:
    #     #     break
    #     # if neg_count >= 150:
    #     #     break
        
    return partition, labels


from sklearn.model_selection import KFold
import numpy as np

def train_val_split_kfolds( folds_n: int = 5, 
                            classes_list: List = [], 
                            shuffle_seed: int = 42) -> Tuple[DefaultDict, Dict]:
    """
    Splits dataset into n folds. 
    Generates two dictionaries required for Torch Generator.
    :param val_ratio: share of validation set
    :return: partition['train'] contains a list of training sample paths
             partition['validation'] contains a list of validation sample paths
             labels contains the associated label for each sample of the dataset

    classes_list = list of ClassesDefinition() objects
    """
    assert len(classes_list) > 0

    kf = KFold(n_splits=folds_n, random_state=shuffle_seed, shuffle=True)

    folds_partions_dict = {}
    labels = {}

    # fill labels
    for clazz in classes_list:
        samples_paths = clazz.get_samples_paths()
        for i, sample_path in enumerate(samples_paths):
            labels[str(sample_path)] = clazz.get_int_label()
    # fill partitions struct
    for i in range(folds_n):
        folds_partions_dict[i] = defaultdict(list)
        folds_partions_dict[i]['train'] = []
        folds_partions_dict[i]['validation'] = []

    # fill folds partitions
    for clazz in classes_list:
        samples_count = clazz.get_samples_count()
        samples_paths = clazz.get_samples_paths()
        random.Random(shuffle_seed).shuffle(samples_paths)

        fold_indx = 0
        for train_index, test_index in kf.split(np.array(range(samples_count))):
            
            partition = folds_partions_dict[fold_indx]
            
            train_X = [samples_paths[i] for i in train_index]
            partition['train'].extend(train_X)
            test_X = [samples_paths[i] for i in test_index]
            partition['validation'].extend(test_X)

            fold_indx = fold_indx + 1
        
    return folds_partions_dict, labels

def train_val_split_strat_kfolds(   folds_n: int = 5, 
                                    classes_list: List = [], 
                                    shuffle_seed: int = 42) -> Tuple[DefaultDict, Dict]:
    """
    Splits dataset into n folds. Takes into account diagnozes in each class to make a trully stratified split.
    Generates two dictionaries required for Torch Generator.
    :param folds_n: number of folds
    :return: partition['train'] contains a list of training sample paths
             partition['validation'] contains a list of validation sample paths
             labels contains the associated label for each sample of the dataset

    classes_list = list of ClassesDefinition() objects
    """

    import random

    def collect_folds_for_path(path, acceptable_diagnoses_names_list, folds_n, shuffle=True, shuffle_seed=shuffle_seed):
        folds_dict = {}
        for i in range(folds_n):
            folds_dict[i] = []

        for diag in os.listdir(path):
        #     print('diag={}'.format(diag))
            if diag.lower() in acceptable_diagnoses_names_list:
                diag_path = os.path.join(path, diag)
    #             print('diag_path={} length={}'.format(diag_path, len(os.listdir(diag_path))))
                f_num = 0
                for diag_sample in os.listdir(diag_path):
                    diag_sample_path = os.path.join(diag_path, diag_sample)
                    folds_dict[f_num].append(diag_sample_path)
                    f_num = f_num + 1
                    if f_num >= folds_n:
                        f_num = 0
        
        if shuffle:
            for i in range(folds_n):
                lst = folds_dict[i]
                random.Random(shuffle_seed).shuffle(lst)
        
        return folds_dict

    def merge_dicts_folds(target_dict, source_dict, folds_n):
        for i in range(folds_n):
            target_dict[i].extend(source_dict[i])

    assert len(classes_list) > 0

    kf = KFold(n_splits=folds_n, random_state=shuffle_seed, shuffle=True)

    folds_partions_dict = {}
    labels = {}

    # fill labels
    for clazz in classes_list:
        samples_paths = clazz.get_samples_paths()
        for i, sample_path in enumerate(samples_paths):
            labels[str(sample_path)] = clazz.get_int_label()
    # fill partitions struct
    for i in range(folds_n):
        folds_partions_dict[i] = defaultdict(list)
        folds_partions_dict[i]['train'] = []
        folds_partions_dict[i]['validation'] = []

    # fill folds partitions
    full_partition_folds_dict = {}
    for i in range(folds_n):
        full_partition_folds_dict[i] = []

    for clazz in classes_list:
        
        clazz_folds_dict = {}
        for i in range(folds_n):
            clazz_folds_dict[i] = []

        for root_folder_path in clazz.root_folders_list: 
            root_folder_dict_folds = \
                collect_folds_for_path( path=str(root_folder_path), 
                                        acceptable_diagnoses_names_list=clazz.diagnoses_names_list, 
                                        folds_n=folds_n)
            merge_dicts_folds(target_dict=clazz_folds_dict, source_dict=root_folder_dict_folds, folds_n=folds_n)
        
        merge_dicts_folds(target_dict=full_partition_folds_dict, source_dict=clazz_folds_dict, folds_n=folds_n)

    # we have dictionary with fold_n -> list of samples
    # now we need to transforma that dict to fold -> {'train': lst, 'validation':lst}
    for i in range(folds_n):
        validation_list = full_partition_folds_dict[i]
        
        train_indexes = list(range(folds_n))
        train_indexes.remove(i)
        train_list = []
        for tr_indx in train_indexes:
            train_list.extend(full_partition_folds_dict[tr_indx])

        random.Random(shuffle_seed).shuffle(validation_list)
        random.Random(shuffle_seed).shuffle(train_list)

        folds_partions_dict[i]['train'] = train_list
        folds_partions_dict[i]['validation'] = validation_list
        
    return folds_partions_dict, labels



if __name__ == '__main__':

    partition, labels = train_val_split(val_ratio=0.1, classes_list=CLASSES_SET)

    # Generators Declaration
    training_set = Dataset(partition['train'], labels, device='cpu')
    training_generator = torch.utils.data.DataLoader(training_set, **PARAMS)

    validation_set = Dataset(partition['validation'], labels, device='cpu')
    validation_generator = torch.utils.data.DataLoader(validation_set, **PARAMS)


    # # Model Training
    # for epoch in range(max_epochs):
    #     # Training
    #     for local_batch, local_labels in training_generator:
    #         # Transfer to GPU
    #         local_batch, local_labels = local_batch.to(device), local_labels.to(device)
    #
    #         # Model computations
    #         [...]
    #
    #     # Validation
    #     with torch.set_grad_enabled(False):
    #         for local_batch, local_labels in validation_generator:
    #             # Transfer to GPU
    #             local_batch, local_labels = local_batch.to(device), local_labels.to(device)
    #
    #             # Model computations
    #             [...]