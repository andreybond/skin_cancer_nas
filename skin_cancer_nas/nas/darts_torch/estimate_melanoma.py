# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys
import logging
import time
from argparse import ArgumentParser
import random
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

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

    # from config import CLASSES_SET
    ROOT_PATHS = [  Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels/checkyourskin'),
                    Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels/loc'),
                    Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels/loc_old_colored')]

    CLS1 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43', 'd03', 'd03.9'], class_name='Melanoma_like_lesions', int_label=0)
    CLS2 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22', 'd81', 'l81.2','l81.4', 'q82.5'], class_name='Pigmented_benign', int_label=1)
    CLS3 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d86.3', 'l21', 'l57', 'l57.0', 'l82', 'l85', 'l85.1', 'l85.5', 'l85.8', 'q80'], class_name='Keratin_lesions', int_label=2)
    CLS4 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c44', 'c46', 'd09'], class_name='Nonmelanoma_skin_cancer', int_label=3)

    CLASSES_SET_1 = [CLS1, CLS2]
    CLASSES_SET_2 = [CLS1, CLS2, CLS3]
    CLASSES_SET_3 = [CLS1, CLS2, CLS3, CLS4]
    CLASSES_SET_4 = [ \
        ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43'], class_name='Melanoma', int_label=0), \
        ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22'], class_name='Nevuss', int_label=1) \
    ]
    CLASSES_SET_5 = [ \
        ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43'], class_name='Melanoma', int_label=0), \
        ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22'], class_name='Nevuss', int_label=1), \
        ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l82'], class_name='Keratin lesion', int_label=2) \
    ]
    CLASSES_SET_6 = [ \
    ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43'], class_name='Melanoma', int_label=0), \
    ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c44'], class_name='NonMelanomaCancer', int_label=1), \
    ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22'], class_name='Nevuss', int_label=2) \
    ]

    layers = [1,2,3,4,5]
    # sets_dict = dict(zip([1,2,3,4,5,6], [CLASSES_SET_1, CLASSES_SET_2, CLASSES_SET_3, CLASSES_SET_4, CLASSES_SET_5, CLASSES_SET_6]))        
    sets_dict = dict(zip([3], [CLASSES_SET_3]))        

    def calculate_f1(args, layer_id, set_id, set_):

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
        data_device = device
        # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        writer = SummaryWriter()

        # layer_id = 1
        # set_id = 4



        partition, labels = data_gen.train_val_split(val_ratio=0.1, classes_list=CLASSES_SET)

        # MEAN = [0.2336, 0.6011, 0.3576, 0.4543]
        # STD = [0.0530, 0.0998, 0.0965, 0.1170]
        # normalize = [
        #     # transforms.Normalize(MEAN, STD)
        # ]
        # train_transform = transforms.Compose(normalize)
        # valid_transform = transforms.Compose(normalize)

        # Generators Declaration
        
        if args.debug == 1:
            debug = True
        else:
            debug = False

        # PARAMS = {'batch_size': 1,
        #         'shuffle': False,
        #         'num_workers': 1}

        # training_set = Dataset(partition['train'], labels, 
        #                         transform=train_transform, 
        #                         device=data_device,
        #                         debug=debug)
        # train_loader = torch.utils.data.DataLoader(training_set, **PARAMS)  #, pin_memory=True)
        # validation_set = Dataset(partition['validation'], labels, 
        #                         transform=valid_transform, 
        #                         device=data_device,
        #                         debug=debug)
        # valid_loader = torch.utils.data.DataLoader(validation_set, **PARAMS)  #, pin_memory=True)

        model_ft = torch.load(args.model_path)
        model_ft.to(device)
        model_ft.eval()

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

        # VALID_CHANNELS = ['r-r', 'ir-r', 'g-g', 'uv-0-r', 'uv-0-g', 'white-g']  #['r', 'ir', 'g', 'uv']
        VALID_CHANNELS = ['r-r', 'ir-r', 'g-g', 'uv-0-g']  #['r', 'ir', 'g', 'uv']
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

        def calc_conf_mat(part, n_classes=2):
            nb_classes = n_classes
            confusion_matrix = torch.zeros(nb_classes, nb_classes)
            with torch.no_grad():
                i_class_list = []
                i_pred_list = []
                for input_path in part:
                    # logger.debug('Processing input_path={}'.format(input_path))
                    inputs, classes = get_input_output_from_path(input_path, labels)

                    inputs = inputs.to(device)
                    classes = classes.to(device)
                    outputs = model_ft(inputs.unsqueeze(0))  #inputs.unsqueeze(0) -- adds first dimension to emulate batch
                    _, preds = torch.max(outputs, 1)

                    i_class = classes.data.cpu().numpy().item()
                    i_pred = preds.data.cpu().numpy()[0].item()
                    i_class_list.append(i_class)
                    i_pred_list.append(i_pred)
                    # # logger.info('   path={}  y_target={}, y_pred={}'.format(input_path, i_class, i_pred))
                    # if i_class != i_pred:
                    #     logger.info('**Sample path={} is class={}, but were classified as {}!'.format(input_path, i_class, i_pred))
                    # # if debug:
                    # #     preds_np = preds.numpy()
                    # #     logger.info('    preds={}'.format(preds_np))
                    for t, p in zip(classes.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
            
            f1 = f1_score(i_class_list, i_pred_list, average='macro')
            return confusion_matrix, f1

        logger.info('Starting estimation.')
        cfmat_val, f1_val = calc_conf_mat(partition['validation'], n_classes)
        cfmat_train, f1_train = calc_conf_mat(partition['train'], n_classes)
        logger.info('Valid: \n' + str(cfmat_val))
        logger.info('Train: \n' + str(cfmat_train))
        logger.info('F1 scores:')
        logger.info('Valid: ' + str(f1_val))
        logger.info('Train: ' + str(f1_train))

        # best_rocauc = 0.
        # for epoch in range(args.epochs):
        #     drop_prob = args.drop_path_prob * epoch / args.epochs
        #     model.drop_path_prob(drop_prob)

        #     # training
        #     train(args, train_loader, model, optimizer, criterion, epoch)

        #     # validation
        #     cur_step = (epoch + 1) * len(train_loader)
        #     roc_auc = validate(args, valid_loader, model, criterion, epoch, cur_step)
        #     best_rocauc = max(best_rocauc, roc_auc)

        #     lr_scheduler.step()

        # logger.info("Final best RocAuc@1 = {:.4%}".format(best_rocauc))
        # torch.save(model, '/mnt/models/darts_retrained_128_nonnormimg_2lrs/model.pt')

    for l in layers:
        for set_id, s in sets_dict.items():
            args = SimpleNamespace(batch_size=1, 
                            model_path="/mnt/models/darts_retrained/6ch_128x128_no_metainfo_registered/NO_IRch_{}lrs_2oct_ClassSet{}_registered/final_model1.pt".format(l, set_id), \
                            debug=1)  # { 0-None | 1-Debug }


            print('Layer-{}, set-{}'.format(l, set_id))
            calculate_f1(args, l, set_id, s)
            print('--------------------------------')

if __name__ == "__main__":

    # args = get_arguments(sys.argv[1:])
    main()