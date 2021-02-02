import sys
import logging
import time
from argparse import ArgumentParser
import os

import torch
import torch.nn as nn

sys.path.append('/mnt')
sys.path.append('/mnt/skin_cancer_nas')
sys.path.append('/mnt/skin_cancer_nas/data/torch_generator')
from skin_cancer_nas.data.torch_generator import generator as data_gen
from skin_cancer_nas.data.torch_generator import base_classes
from skin_cancer_nas.data.torch_generator import config
#from skin_cancer_nas.data.torch_generator import preprocessor

from sklearn.metrics import f1_score

import datasets
from model_melanoma import Combine, CNN
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback
from nni.nas.pytorch.darts import DartsTrainer
# from val_ratio
from utils import f1_loss
from utils import f1
from base_classes import Dataset
from torchvision import transforms

# from config import CLASSES_SET



# logger.setLevel(logging.DEBUG)
# # create file handler which logs even debug messages
# fh = logging.FileHandler('error.log')
# fh.setLevel(logging.ERROR)
# # create formatter and add it to the handlers
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
# # add the handlers to logger
# logger.addHandler(fh)

from types import SimpleNamespace
from config import ClassesDefinition
from pathlib import Path


def run(args):

    checkpoints_folder = args.checkpoints_folder
    assert checkpoints_folder is not None

    os.makedirs('/mnt/models/multi_search_14Dec2020_128x128_4ch_9ch/'+checkpoints_folder, exist_ok=True)

    if args.classes_set is not None:
        classes = args.classes_set
    else:
        classes = CLASSES_SET

    logging.basicConfig(filename='/mnt/models/multi_search_14Dec2020_128x128_4ch_9ch/'+checkpoints_folder+'/search_log.txt',
                        filemode='a',  # a - append
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger('nni')

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.cuda_id))  # you can continue going on here, like cuda:1 cuda:2....etc. 
        logger.info("Running on the GPU")
    else:
        device = torch.device("cpu")
        logger.info("Running on the CPU")

    # dataset_train, dataset_valid = datasets.get_dataset("cifar10")
    partition, labels = data_gen.train_val_split(val_ratio=0.1, classes_list=CLASSES_SET)

    MEAN = [0.2336, 0.6011, 0.3576, 0.4543]
    STD = [0.0530, 0.0998, 0.0965, 0.1170]
    normalize = [
        # transforms.Normalize(MEAN, STD),

    ]
    train_transform = transforms.Compose(normalize)
    valid_transform = transforms.Compose(normalize)

    logger.info('-valid_channels={}'.format(args.valid_channels))

    # Generators Declaration
    training_set = Dataset(partition['train'], labels, 
                           transform=train_transform, 
                           device=device,
                           valid_channels=args.valid_channels)
    training_generator = torch.utils.data.DataLoader(training_set, **data_gen.PARAMS, pin_memory=True)
    validation_set = Dataset(partition['validation'], labels, 
                           transform=valid_transform, 
                           device=device,
                           valid_channels=args.valid_channels)
    validation_generator = torch.utils.data.DataLoader(validation_set, **data_gen.PARAMS, pin_memory=True)
    
    #Model declaration                
    cnn_model = CNN(input_size=320, 
                    in_channels=len(args.valid_channels), 
                    channels=args.channels, 
                    n_classes=len(set(labels.values())), 
                    n_layers=args.layers)
    # combined_model = Combine(cnn_model)
    cnn_model.to(device)
    #cnn_model = cnn_model.cuda()
    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(cnn_model.parameters(recurse=True), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)

    # def f1_metric(out, target, avg_policy='macro'):
    #     output = out.cpu().detach().numpy()
    #     return f1_score(target.cpu(), output.argmax(-1), average=avg_policy)

    trainer = DartsTrainer(cnn_model,
                           loss=criterion,
                        #    loss=f1_loss,
                        #    metrics=lambda output, target: utils.accuracy(output, target, topk=(1,)),
                        #    metrics=lambda output, target: f1_score(output.cpu().detach().numpy(), target.cpu().detach().numpy()),
                           metrics=lambda output, target: f1(y_true=target, y_pred=output, is_training=True),
                           optimizer=optim,
                           num_epochs=args.epochs,
                           dataset_train=training_set,
                           dataset_valid=validation_set,
                           batch_size=args.batch_size,
                           log_frequency=args.log_frequency,
                           unrolled=args.unrolled,
                           workers=0,
                           callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("/mnt/models/multi_search_14Dec2020_128x128_4ch_9ch/"+checkpoints_folder)])
    
    if args.visualization:
        trainer.enable_visualization()
    trainer.train()

    logger.info('Done!')


if __name__ == "__main__":

    # ROOT_PATHS = [  Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels/checkyourskin'),
    #             Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels/loc'),
    #             Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels/loc_old_colored')]

    ROOT_PATHS = [  Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/checkyourskin'),
                Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/loc_ALL_WHITE'),
                Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/loc_old_colored'),
                Path('/mnt/data/interim/_melanoma_20200728_2__checkyourskin_c43_NoDuplicates_14Nov2020')]

    # CLASSES_SET = [ ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43', 'd03', 'd03.9'], class_name='Melanoma_like_lesions', int_label=0),
    #             ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22', 'd81', 'l81.2','l81.4', 'q82.5'], class_name='Pigmented_benign', int_label=1),
    #             ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d86.3', 'l21', 'l57', 'l57.0', 'l82', 'l85', 'l85.1', 'l85.5', 'l85.8', 'q80'], class_name='Keratin_lesions', int_label=2)]

    CLS1 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43', 'd03', 'd03.9'], class_name='Melanoma_like_lesions', int_label=0)
    CLS2 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22', 'l81.2','l81.4', 'q82.5'], class_name='Pigmented_benign', int_label=1)  #'d81', 
    CLS3 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d86.3', 'l21', 'l57', 'l57.0', 'l82', 'l85', 'l85.1', 'l85.5', 'l85.8', 'q80'], class_name='Keratin_lesions', int_label=2)
    CLS4 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c44', 'c46', 'd09'], class_name='Nonmelanoma_skin_cancer', int_label=3)
    CLS5 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['a63', 'd18', 'd21.9', 'd48', 'l92', 'l94.2', 'l98.8', 'pxe', 'b07', 'ada', 'l57.9', 'l98.9'], class_name='Other', int_label=4)

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
    CLASSES_SET_8 = [CLS1, CLS2, CLS3, CLS4, CLS5]

    CLASSES_SET = CLASSES_SET_8

    # parser = ArgumentParser("darts")
    # parser.add_argument("--layers", default=5, type=int)
    # parser.add_argument("--batch-size", default=4, type=int)
    # parser.add_argument("--log-frequency", default=10, type=int)
    # parser.add_argument("--epochs", default=50, type=int)
    # parser.add_argument("--channels", default=16, type=int)
    # parser.add_argument("--unrolled", default=False, action="store_true")
    # parser.add_argument("--visualization", default=False, action="store_true")
    # parser.add_argument("--checkpoints_folder", default=None)
    # parser.add_argument("--cuda_id", default=0, type=int)
    # #parser.add_argument("--valid_channels", default=['r-r', 'ir-r', 'g-g', 'uv-r', 'uv-g', 'white-g', 'white-r'], type=int)
    #parser.add_argument("--classes_set", default=CLASSES_SET, type=int)
    # args = parser.parse_args()

    cuda_id_ = 0
    do_visualize = True
    # CHANNELS = ['r-r', 'ir-r', 'g-g', 'uv-0-r', 'uv-0-g', 'white-g']
    # CHANNELS = ['r-r', 'ir-r', 'g-g', 'uv-0-r', 'uv-0-g', 'white-g', 'white-r']
    # CHANNELS = ['r-r', 'ir-r', 'g-g', 'uv-0-r', 'uv-0-g']
    CHANNELS = ['r-r', 'ir-r', 'g-g', 'uv-0-g']
    _batch_size = 4
    arch_channels = 9
    runs_args = [
        SimpleNamespace(layers=1, batch_size=_batch_size, log_frequency=10, epochs=50, channels=arch_channels, unrolled=False, visualization=do_visualize, \
                        checkpoints_folder='search_1lr_4ch_5classes_9innerCh', cuda_id=cuda_id_, \
                        valid_channels=CHANNELS, \
                        classes_set=CLASSES_SET),

        SimpleNamespace(layers=2, batch_size=_batch_size, log_frequency=10, epochs=50, channels=arch_channels, unrolled=False, visualization=do_visualize, \
                        checkpoints_folder='search_2lr_4ch_5classes_9innerCh', cuda_id=cuda_id_, \
                        valid_channels=CHANNELS, \
                        classes_set=CLASSES_SET),
        
        SimpleNamespace(layers=3, batch_size=_batch_size, log_frequency=10, epochs=50, channels=8, unrolled=False, visualization=do_visualize, \
                        checkpoints_folder='search_3lr_4ch_5classes_9innerCh', cuda_id=cuda_id_, \
                        valid_channels=CHANNELS, \
                        classes_set=CLASSES_SET),

        SimpleNamespace(layers=4, batch_size=_batch_size, log_frequency=10, epochs=50, channels=arch_channels, unrolled=False, visualization=do_visualize, \
                        checkpoints_folder='search_4lr_4ch_5classes_9innerCh', cuda_id=cuda_id_, \
                        valid_channels=CHANNELS, \
                        classes_set=CLASSES_SET),

        SimpleNamespace(layers=5, batch_size=_batch_size, log_frequency=10, epochs=50, channels=arch_channels, unrolled=False, visualization=do_visualize, \
                        checkpoints_folder='search_5lr_4ch_5classes_9innerCh', cuda_id=cuda_id_, \
                        valid_channels=CHANNELS, \
                        classes_set=CLASSES_SET)#,

        # SimpleNamespace(layers=6, batch_size=_batch_size, log_frequency=10, epochs=50, channels=arch_channels, unrolled=False, visualization=do_visualize, \
        #                 checkpoints_folder='search_6lr_5ch_5classes_16innerCh', cuda_id=cuda_id_, \
        #                 valid_channels=CHANNELS, \
        #                 classes_set=CLASSES_SET),

        # SimpleNamespace(layers=7, batch_size=_batch_size, log_frequency=10, epochs=50, channels=arch_channels, unrolled=False, visualization=do_visualize, \
        #                 checkpoints_folder='search_7lr_5ch_5classes_16innerCh', cuda_id=cuda_id_, \
        #                 valid_channels=CHANNELS, \
        #                 classes_set=CLASSES_SET)
    ]

    for args in runs_args:
        run(args)

