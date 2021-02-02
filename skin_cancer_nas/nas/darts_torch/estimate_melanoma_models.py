
'''
This file is for generation of model outputs and true predictions for the given model and dataset.
It is applicable to XFold files (10-fold cross validation series) as it is looking for log file 
and records of train/validation dataset folders.

Output of the script is 2 files: 
-train_pred.csv
-validation_pred.csv

Each of the files contain columns: 
folder_path, y_true, y_pred, logit0, logit1, ... ,logitN.

So these columns can be used for F1 , ConfMatrix and other scores estimation.
Logits outputs can be used for thresholds tuning experiments.
'''

import sys
import logging
import time
from argparse import ArgumentParser
import random
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import json
from tqdm import tqdm
import pandas as pd

sys.path.append('/mnt')
sys.path.append('/mnt/skin_cancer_nas')
sys.path.append('/mnt/skin_cancer_nas/data/torch_generator')
from skin_cancer_nas.data.torch_generator import generator as data_gen
from skin_cancer_nas.data.torch_generator import base_classes
from skin_cancer_nas.data.torch_generator import config

import cv2
import datasets
import utils
from model_melanoma import CNN
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.utils import AverageMeter
from torchvision import transforms
from base_classes import Dataset
from config import ClassesDefinition
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import numpy as np
from pathlib import Path
from types import SimpleNamespace


logger = logging.getLogger('nni')

# def get_arguments(args):
#     parser = ArgumentParser("darts")
#     parser.add_argument("--batch_size", default=1, type=int)
#     parser.add_argument("--model_path", default="/mnt/models/darts_retrained/6ch_128x128_no_metainfo_registered/{}lrs_2oct_ClassSet{}_registered/final_model1.pt".format(layer_id, set_id))
#     parser.add_argument("--debug", default=1, type=int)  # { 0-None | 1-Debug }

#     return parser.parse_args(args)

def main() -> None:

    def load_partition_from_log(model_path, device):
        
        part_dict = {}
        log_file_path = args.model_path + '/model1.log.txt'
        with open(log_file_path) as fp: 
            for line in fp: 
                if "partition->defaultdict(<class 'list'>, " in line:
                    part_str = line.split("partition->defaultdict(<class 'list'>, ")[1][:-2]
                    part_str = part_str.replace("'", '"')
                    part_dict = json.loads(part_str)

        model_ft = torch.load(args.model_path + '/final_model1.pt')
        model_ft.to(device)
        model_ft.eval()

        return part_dict, model_ft

    def calculate_logits_and_predictions(args, layer_id, set_id, set_):

        CLASSES_SET = set_
        n_classes = len(CLASSES_SET)

        def seed_torch(seed=42):
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        seed_torch()

        device = torch.device("cpu")  #
        # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        writer = SummaryWriter()

        # layer_id = 1
        # set_id = 4


        partition, model_ft = load_partition_from_log(args.model_path, device)
        _, labels = data_gen.train_val_split(val_ratio=0.1, classes_list=CLASSES_SET)

        if args.debug == 1:
            debug = True
        else:
            debug = False

        def get_input_output_from_path(sample_path, labels):
            '''
            Generates one sample of data
            '''
            try:
                # logger.info('Sample path={}'.format(sample_path))
                x_out = _convert_img_to_array(sample_path)
                x_out = x_out.astype('float32')
                x_out /= 255

                X = torch.as_tensor(x_out)
                y = labels[str(sample_path)]
                
                X = X.to(device)
                y = torch.tensor(y).to(device)
                return X, y
            except Exception as e: 
                logger.info('Something went wrong at path={}, e={}'.format(sample_path, e))

        # VALID_CHANNELS = ['r-r', 'ir-r', 'g-g', 'uv-0-r', 'uv-0-g', 'white-g', 'white-r']  #['r', 'ir', 'g', 'uv']
        # VALID_CHANNELS = ['r-r', 'ir-r', 'g-g', 'uv-0-g']  #['r', 'ir', 'g', 'uv']
        # VALID_CHANNELS = ['r-r', 'ir-r', 'g-g', 'uv-0-r', 'uv-0-g']
        VALID_CHANNELS = ['r-r', 'ir-r', 'g-g', 'uv-0-g']
        IMG_WIDTH = 128
        IMG_HEIGHT = 128
        VALUE_MISSING = 0
        MISSING_CH_IMG = np.ones((IMG_HEIGHT, IMG_WIDTH)) * VALUE_MISSING

        def _convert_img_to_array(sample_path):
            'Converts n grayscale images to 3D array with n channels'
            x_array = []
            for channel in VALID_CHANNELS:
                channel_path = os.path.join(sample_path, channel)
                if not os.path.exists(channel_path):
                    x_array.append(MISSING_CH_IMG)
                    continue
                image = os.listdir(channel_path)
                if not image:
                    x_array.append(MISSING_CH_IMG)
                    continue
                full_image_path = os.path.join(channel_path, image[0])
                img = cv2.imread(full_image_path, flags=cv2.IMREAD_GRAYSCALE)
                if img is None:
                    x_array.append(MISSING_CH_IMG)
                    continue
                else:
                    shape = img.shape
                    # logger.info(shape)
                    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
                
                x_array.append(img)

            return np.stack(x_array, axis=0)

        def calc_logits_and_preds(partition, n_classes=2):
            nb_classes = n_classes

            i_true_list = []
            i_pred_list = []
            i_out_list = []

            with torch.no_grad():
                for input_path in tqdm(partition):
                    # logger.debug('Processing input_path={}'.format(input_path))
                    try:
                        input_path2 = input_path#.replace('_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels',
                                                #        '_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED')

                        # inputs, classes = get_input_output_from_path(input_path, labels)
                        inputs, classes = get_input_output_from_path(input_path2, labels)

                        inputs = inputs.to(device)
                        classes = classes.to(device)
                        outputs = model_ft(inputs.unsqueeze(0))  #inputs.unsqueeze(0) -- adds first dimension to emulate batch
                        _, preds = torch.max(outputs, 1)

                        i_true = classes.data.cpu().numpy().item()
                        i_pred = preds.data.cpu().numpy()[0].item()
                        i_outputs = outputs.data.cpu().numpy()
                        
                        # we use batch_size == 1, hence will be adding values to these lists one by one
                        i_true_list.append(i_true)
                        i_pred_list.append(i_pred)
                        i_out_list.append(i_outputs)
                    except:
                        continue  # _MAN_CORRECTED dataset deleted some bad samples, thus FileNotFound could be encountered

            return i_true_list, i_pred_list, i_out_list
        
        def save_to_file(prefix, true_list, pred_list, outs_list, folder_path):
            file_path = folder_path + '/' + prefix + '_logits_predictions.csv'

            logits_count = outs_list[0].shape[1]

            columns = ['true_value', 'pred_value']
            for i in range(logits_count):
                columns.append('logit_{}'.format(i))

            data_dict = {}
            for col in columns:
                data_dict[col] = []

            # create data frame
            df = pd.DataFrame(data_dict)

            for true_val, pred_val, outs_val in zip(true_list, pred_list, outs_list):
                row_list = [true_val, pred_val]
                logits_list = outs_val.reshape(outs_val.shape[1]).tolist()
                row_list.extend(logits_list)
                
                df = df.append(pd.Series(row_list, index=df.columns), ignore_index=True)
            
            df.to_csv(file_path)

        logger.info('Starting estimation...')
        
        i_true_list_val, i_pred_list_val, i_out_list_val = calc_logits_and_preds(partition['validation'], n_classes)
        i_true_list_train, i_pred_list_train, i_out_list_train = calc_logits_and_preds(partition['train'], n_classes)

        save_to_file('train7ch', i_true_list_train, i_pred_list_train, i_out_list_train, args.model_path)
        save_to_file('validation7ch', i_true_list_val, i_pred_list_val, i_out_list_val, args.model_path)

        logger.info('Finished estimation.')


    # Now run this script for a set of folders / models to generate 
    # train_pred.csv, validation_pred.csv files for each folder.
    # These files can then be used in the jupyter notebooks.

    # from config import CLASSES_SET
    # ROOT_PATHS = [  Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels/checkyourskin'),
    #                 Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels/loc'),
    #                 Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels/loc_old_colored')]
    # ROOT_PATHS = [  Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/checkyourskin'),
    #             Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/loc'),
    #             Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/loc_old_colored')]

    # ROOT_PATHS = [  Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/checkyourskin'),
    #             Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/loc'),
    #             Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/loc_old_colored'),
    #             Path('/mnt/data/interim/_melanoma_20200728_2__checkyourskin_c43_NoDuplicates')]

    ROOT_PATHS = [  Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/checkyourskin'),
                    Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/loc_ALL_WHITE'),
                    Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/loc_old_colored'),
                    Path('/mnt/data/interim/_melanoma_20200728_2__checkyourskin_c43_NoDuplicates_14Nov2020')]

    CLS1 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43', 'd03', 'd03.9'], class_name='Melanoma_like_lesions', int_label=0)
    CLS2 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22', 'l81.2','l81.4', 'q82.5'], class_name='Pigmented_benign', int_label=1)  # 'd81' -> should be l21
    CLS3 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d86.3', 'l21', 'l57', 'l57.0', 'l82', 'l85', 'l85.1', 'l85.5', 'l85.8', 'q80'], class_name='Keratin_lesions', int_label=2)
    CLS4 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c44', 'c46', 'd09'], class_name='Nonmelanoma_skin_cancer', int_label=3)
    CLS5 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['a63', 'd18', 'd21.9', 'd48', 'l92', 'l94.2', 'l98.8', 'pxe', 'b07', 'ada', 'l57.9', 'l98.9'], class_name='Other', int_label=4)

    # CLASSES_SET_1 = [CLS1, CLS2]
    # CLASSES_SET_2 = [CLS1, CLS2, CLS3]
    # CLASSES_SET_3 = [CLS1, CLS2, CLS3, CLS4]
    CLASSES_SET_4 = [ \
        ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43'], class_name='Melanoma', int_label=0), \
        ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22'], class_name='Nevuss', int_label=1) \
    ]
    # CLASSES_SET_5 = [ \
    #     ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43'], class_name='Melanoma', int_label=0), \
    #     ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22'], class_name='Nevuss', int_label=1), \
    #     ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l82'], class_name='Keratin lesion', int_label=2) \
    # ]
    # CLASSES_SET_6 = [ \
    #     ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43'], class_name='Melanoma', int_label=0), \
    #     ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c44'], class_name='NonMelanomaCancer', int_label=1), \
    #     ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22'], class_name='Nevuss', int_label=2) \
    # ]

    CLASSES_SET_8 = [CLS1, CLS2, CLS3, CLS4, CLS5]

    CLS11 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43'], class_name='Melanoma_like_lesions1', int_label=0)
    CLS12 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d03'], class_name='Melanoma_like_lesions2', int_label=1)
    CLS13 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d03.9'], class_name='Melanoma_like_lesions3', int_label=2)
    CLS21 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22'], class_name='Pigmented_benign1', int_label=3)   #'d81', 
    CLS22 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l81.2'], class_name='Pigmented_benign2', int_label=4)   #'d81', 
    CLS23 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l81.4'], class_name='Pigmented_benign3', int_label=5)   #'d81', 
    CLS24 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=[ 'q82.5'], class_name='Pigmented_benign4', int_label=6)   #'d81', 
    CLS31 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d86.3'], class_name='Keratin_lesions1', int_label=7)
    CLS32 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l21'], class_name='Keratin_lesions2', int_label=8)
    CLS33 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l57'], class_name='Keratin_lesions3', int_label=9)
    CLS34 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l57.0'], class_name='Keratin_lesions4', int_label=10)
    CLS35 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l82'], class_name='Keratin_lesions5', int_label=11)
    CLS36 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l85'], class_name='Keratin_lesions6', int_label=12)
    CLS37 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l85.1'], class_name='Keratin_lesions7', int_label=13)
    CLS38 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l85.5'], class_name='Keratin_lesions8', int_label=14)
    CLS39 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l85.8'], class_name='Keratin_lesions9', int_label=15)
    CLS310 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['q80'], class_name='Keratin_lesions10', int_label=16)
    CLS41 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c44'], class_name='Nonmelanoma_skin_cancer1', int_label=17)
    CLS42 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c46'], class_name='Nonmelanoma_skin_cancer2', int_label=18)
    CLS43 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d09'], class_name='Nonmelanoma_skin_cancer3', int_label=19)
    CLS51 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['a63'], class_name='Other1', int_label=20)
    CLS52 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d18'], class_name='Other2', int_label=21)
    CLS53 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d21.9'], class_name='Other3', int_label=22)
    CLS54 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d48'], class_name='Other4', int_label=23)
    CLS55 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l92'], class_name='Other5', int_label=24)
    CLS56 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l94.2'], class_name='Other6', int_label=25)
    CLS57 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l98.8'], class_name='Other7', int_label=26)
    CLS58 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['pxe'], class_name='Other8', int_label=27)
    CLS59 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['b07'], class_name='Other9', int_label=28)
    CLS510 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['ada'], class_name='Other10', int_label=29)
    CLS511 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l57.9'], class_name='Other11', int_label=30)
    CLS512 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l98.9'], class_name='Other12', int_label=31)

    CLASSES_SET_9 = [CLS11, CLS12, CLS13,
                    CLS21, CLS22, CLS23, CLS24,
                    CLS31, CLS32, CLS33, CLS34, CLS35, CLS36, CLS37, CLS38, CLS39, CLS310,
                    CLS41, CLS42, CLS43,
                    CLS51, CLS52, CLS53, CLS54, CLS55, CLS56, CLS57, CLS58, CLS59, CLS510, CLS511, CLS512]

    layers = [1,2,3,4,5]
    folds = [0,1,2,3,4]
    # sets_dict = dict(zip([1,2,3,4,5,6], [CLASSES_SET_1, CLASSES_SET_2, CLASSES_SET_3, CLASSES_SET_4, CLASSES_SET_5, CLASSES_SET_6]))        
    sets_dict = dict(zip([9], [CLASSES_SET_9]))        
    
    for l in layers:
        for set_id, s in sets_dict.items():
            for fold_n in folds:
                args = SimpleNamespace(batch_size=1,
                                       model_path="/mnt/models/darts_retrained/4ch_128x128_12inchannel_no_metainfo_registered_5Fold_newC43_augC43_v1_21Dec2020_32classes/XV2_SGD_orig_02DropChannel_{}lrs_15Dec_newC43NoDup_ClassSet{}_ManCorected_registered_350epochs_fold-{}".format(l, set_id, fold_n),
                                       debug=1)  # { 0-None | 1-Debug }

                print('Layer-{}, set-{}, fold_n-{}'.format(l, set_id, fold_n))
                calculate_logits_and_predictions(args, l, set_id, s)
                print('--------------------------------')

if __name__ == "__main__":

    # args = get_arguments(sys.argv[1:])
    main()