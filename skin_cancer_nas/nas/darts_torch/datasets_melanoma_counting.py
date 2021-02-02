"""
This script utilizes defined PyTorch data loader to caclulate statistics required for images normalization.
It calculates std and mean for each separate channel. 
These values are then hardcoded in the data transformer normalization step.
"""
import sys
import os
from pathlib import Path
import operator

sys.path.append('/mnt')
sys.path.append('/mnt/skin_cancer_nas')
sys.path.append('/mnt/skin_cancer_nas/data/torch_generator')

# from skin_cancer_nas.data.torch_generator import generator as data_gen  
# from skin_cancer_nas.data.torch_generator import base_classes
# from skin_cancer_nas.data.torch_generator import config
# from skin_cancer_nas.data.torch_generator import preprocessor

import torch
from torchvision import transforms

# from base_classes import Dataset
import logging
logger = logging.getLogger('nni')

if __name__ == '__main__':
    logger.info('os.getcwd() = ' + os.getcwd())

    # # ROOT_PATHS = [  Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels/checkyourskin'),
    # #             Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels/loc'),
    # #             Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels/loc_old_colored')]

    # # ROOT_PATHS = [  Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/checkyourskin'),
    # #             Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/loc'),
    # #             Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/loc_old_colored')]

    # # ROOT_PATHS = [  Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/checkyourskin'),
    # #             Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/loc'),
    # #             Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/loc_old_colored'),
    # #             Path('/mnt/data/interim/_melanoma_20200728_2__checkyourskin_c43_NoDuplicates')]


    # # overall
    # used_diagnozes = ['c43', 'd03', 'd03.9', 
    #                   'd22', 'd81', 'l81.2','l81.4', 'q82.5',
    #                   'd86.3', 'l21', 'l57', 'l57.0', 'l82', 'l85', 'l85.1', 'l85.5', 'l85.8', 'q80',
    #                   'c44', 'c46', 'd09',
    #                   'a63', 'd18', 'd21.9', 'd48', 'l92', 'l94.2', 'l98.8', 'pxe', 'b07', 'ada', 'l57.9', 'l98.9']

    ROOT_PATHS = [  Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/checkyourskin'),
                Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/loc_ALL_WHITE'),
                Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/loc_old_colored'),
                Path('/mnt/data/interim/_melanoma_20200728_2__checkyourskin_c43_NoDuplicates_14Nov2020')]

    used_diagnozes =   ['c43', 'd03', 'd03.9',
                        'd22', 'l81.2','l81.4', 'q82.5', 
                        'd86.3', 'l21', 'l57', 'l57.0', 'l82', 'l85', 'l85.1', 'l85.5', 'l85.8', 'q80',
                        'c44', 'c46', 'd09',
                        'a63', 'd18', 'd21.9', 'd48', 'l92', 'l94.2', 'l98.8', 'pxe', 'b07', 'ada', 'l57.9', 'l98.9']

    white_counter = {}
    diagnoze_counts = {}
    for root in ROOT_PATHS:
        for diag_dir in next(os.walk(root))[1]:
            path = os.path.join(root, diag_dir)
            diag_count = len([i for i in os.listdir(path) if os.path.isdir( os.path.join(path, i) )])
            if diag_dir not in diagnoze_counts:
                diagnoze_counts[diag_dir] = 0
            diagnoze_counts[diag_dir] = diagnoze_counts[diag_dir] + diag_count
        
        for root, dirs, files in os.walk(root):
            for filename in files:
                if filename.lower().startswith('white') and any(diag in root.lower() for diag in used_diagnozes):
                    if filename not in white_counter:
                        white_counter[filename] = 0
                    white_counter[filename] = white_counter[filename] + 1

    print('white_counter:')
    print(white_counter)
    print(' ')
    
    diagnoze_counts = dict( sorted(diagnoze_counts.items(), key=operator.itemgetter(1),reverse=True))
    print(diagnoze_counts)