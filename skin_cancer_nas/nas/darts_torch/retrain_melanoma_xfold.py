# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys

import time
from argparse import ArgumentParser


import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.append('/mnt')
sys.path.append('/mnt/skin_cancer_nas')
sys.path.append('/mnt/skin_cancer_nas/data/torch_generator')
from skin_cancer_nas.data.torch_generator import generator as data_gen
from skin_cancer_nas.data.torch_generator import base_classes
from skin_cancer_nas.data.torch_generator.config import *
from skin_cancer_nas.data.torch_generator import config

from types import SimpleNamespace
from config import ClassesDefinition
from pathlib import Path

from base_classes import Dataset
import datasets
import utils
from model_melanoma import CNN
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.utils import AverageMeter
from torchvision import transforms
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import os

seed = 42
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


# def get_arguments(args):
#     parser = ArgumentParser("darts")
#     parser.add_argument("--layers", default=5, type=int)
#     parser.add_argument("--batch-size", default=4, type=int)
#     parser.add_argument("--log-frequency", default=10, type=int)
#     parser.add_argument("--epochs", default=750, type=int)
#     parser.add_argument("--aux-weight", default=0.4, type=float)
#     parser.add_argument("--drop-path-prob", default=0.2, type=float)
#     parser.add_argument("--workers", default=16)
#     parser.add_argument("--grad-clip", default=5., type=float)
#     parser.add_argument("--arc-checkpoint", default="/mnt/c43_d22_6chimg_only_checkpoints_128_nonnormimg_5lrs_16ch/epoch_49.json")    
#     parser.add_argument("--cuda_id", default=0, type=int)
#     parser.add_argument("--valid_channels", default=None)
#     parser.add_argument("--target_folder_name", default=None, type=str)
#     parser.add_argument("--classes_set", default=None)
#     parser.add_argument("--logger_name", default='nni_darts')

#     return parser.parse_args(args)


class shades_of_gray(object):
   
   #     Parameters
   #    ----------
   #   img: 2D numpy array
   #         The original image with format of (h, w, c)
   #     power: int
   #         The degree of norm, 6 is used in reference paper
   # 
     
    
    def __call__(self, img):
        """
        :param img: PIL): Image 

        :return: Normalized image
        """
        img = numpy.asarray(img)
        img_dtype = img.dtype

        power = 6
        extra = 6

        img = img.astype('float32')
        img_power = numpy.power(img, power)
        rgb_vec = numpy.power(numpy.mean(img_power, (0, 1)), 1 / power)
        rgb_norm = numpy.power(numpy.sum(numpy.power(rgb_vec, extra)), 1 / extra)
        rgb_vec = rgb_vec / rgb_norm
        rgb_vec = 1 / (rgb_vec * numpy.sqrt(3))
        img = numpy.multiply(img, rgb_vec)
        img = img.astype(img_dtype)

        return Image.fromarray(img)

    def __repr__(self):
        return self.__class__.__name__+'()'

import random

class RandomRot(object):
    
    '''
    Randomly rotates +90 / -90 degrees (or not rotates) passed in 3D tensor
    '''

    def __call__(self, sample):
        """
        :param sample: torch tensor 

        :return: randomly rotated for 90, 180, 270 or 0 times
        """
        k = random.randint(0, 2) - 1
        if k != 0:
            sample = torch.rot90(sample, k, [1, 2])
        return sample

class RandomFlip(object):
    
    def __init__(self, horizontal=True, prob_threshold=0.5):
        self.prob_threshold = prob_threshold
        self.horizontal = horizontal

    def __call__(self, sample):
        """
        :param sample: torch tensor 

        :return: randomly horizontally/vertically flipped torch 3D tensor
        """
        flip_prob = random.uniform(0, 1)
        if flip_prob > self.prob_threshold:
            if self.horizontal:
                sample = torch.flip(sample, [2])  # vertical flip of 3D tensor
            else:
                sample = torch.flip(sample, [1]) # horizontal flip of 3D tensor
        return sample

import logging

def prepare_train_val_generators(partition, labels):
    # MEAN = [0.2336, 0.6011, 0.3576, 0.4543]
    # STD = [0.0530, 0.0998, 0.0965, 0.1170]
    normalize = [
        # # transforms.Normalize(MEAN, STD)
        # transforms.ToPILImage(),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(degrees=(-90, 90)),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.ToTensor(),
        RandomFlip(horizontal=True, prob_threshold=0.5),
        RandomRot(),
        RandomFlip(horizontal=False, prob_threshold=0.5)
    ]
    train_transform = transforms.Compose(normalize)
    valid_transform = transforms.Compose([])

    # Generators Declaration
    data_device = torch.device("cpu")
    # data_device = device

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    training_set = Dataset( partition['train'], 
                            labels, 
                            transform=train_transform, 
                            device=data_device,
                            valid_channels=args.valid_channels,
                            channels_to_zero_out=args.channels_to_zero_out)
    train_loader = torch.utils.data.DataLoader( training_set, 
                                                **data_gen.PARAMS, 
                                                pin_memory=True, 
                                                worker_init_fn=_init_fn)

    validation_set = Dataset(partition['validation'], 
                            labels, 
                            transform=valid_transform, 
                            device=data_device,
                            valid_channels=args.valid_channels,
                            channels_to_zero_out=args.channels_to_zero_out)
    valid_loader = torch.utils.data.DataLoader( validation_set, 
                                                **data_gen.PARAMS, 
                                                pin_memory=True, 
                                                worker_init_fn=_init_fn)

    return train_loader, valid_loader

def save_model(path_to_folder, file_name, model):
        os.makedirs(path_to_folder, exist_ok=True)
        torch.save(model, os.path.join(path_to_folder, file_name))

def calc_conf_mat(loader, n_classes, device, model):
        nb_classes = n_classes
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        with torch.no_grad():
            for i, (inputs, classes) in enumerate(loader):
                inputs = inputs.to(device)
                classes = classes.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        return confusion_matrix

def main(args: ArgumentParser) -> None:

    def train(config, train_loader, model, optimizer, criterion, epoch):
        f1s = AverageMeter("f1")
        precisions = AverageMeter("precision")
        recalls = AverageMeter("recall")
        rocaucs = AverageMeter("roc-auc")
        losses = AverageMeter("losses")

        cur_step = epoch * len(train_loader)
        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %d LR %.6f", epoch, cur_lr)
        writer.add_scalar("lr", cur_lr, global_step=cur_step)

        model.train()

        for step, (x, y) in enumerate(train_loader):
            # logger.info('X.size() = {}'.format(x.size()))
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            bs = x.size(0)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            # if config.aux_weight > 0.:
            #     loss += config.aux_weight * criterion(aux_logits, y)
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            losses.update(loss.item(), bs)
            
            output = logits.cpu().detach().numpy()
            f1 = f1_score(y.cpu(), output.argmax(-1), average='macro')
            f1s.update(f1, bs)

            recall = recall_score(y.cpu(), output.argmax(-1), average='macro')
            recalls.update(recall, bs)

            precision = precision_score(y.cpu(), output.argmax(-1), average='macro')
            precisions.update(precision, bs)

            # rocauc = roc_auc_score(y.cpu(), np.amax(output, axis=1))
            # roaucs.update(rocauc, bs)

            writer.add_scalar("loss/train", loss.item(), global_step=cur_step)
            writer.add_scalar("f1/train", f1, global_step=cur_step)
            writer.add_scalar("precision/train", precision, global_step=cur_step)
            writer.add_scalar("recall/train", recall, global_step=cur_step)
            # writer.add_scalar("rocauc/train", rocauc, global_step=cur_step)

            if step % config.log_frequency == 0 or step == len(train_loader) - 1:
                logger.info(
                    "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "F1 ({f1s.avg:.1%}) Precision({precisions.avg:.1%}) Recall({recalls.avg:.1%}) )".format( # RocAuc({rocaucs.avg:.1%}
                        epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
                        f1s=f1s, precisions=precisions, recalls=recalls, )) # rocaucs=rocaucs

            cur_step += 1

        logger.info("Train: [{:3d}/{}] Final (avg) F1@1 {:.4%} Precisions {:.4%} Recalls {:.4%} ".format( #RocAucs {:.4%}
            epoch + 1, config.epochs, f1s.avg, precisions.avg, recalls.avg, )) # rocaucs.avg

    def validate(config, valid_loader, model, criterion, epoch, cur_step):
        f1s = AverageMeter("f1")
        precisions = AverageMeter("precision")
        recalls = AverageMeter("recall")
        rocaucs = AverageMeter("roc-auc")
        losses = AverageMeter("losses")

        model.eval()

        with torch.no_grad():
            for step, (X, y) in enumerate(valid_loader):
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                bs = X.size(0)

                logits = model(X)
                loss = criterion(logits, y)

                # f1_val = utils.f1(logits, y)
                # losses.update(loss.item(), bs)
                # f1.update(accuracy["f1"], bs)

                output = logits.cpu().detach().numpy()
                f1 = f1_score(y.cpu(), output.argmax(-1), average='macro')
                f1s.update(f1, bs)

                recall = recall_score(y.cpu(), output.argmax(-1), average='macro')
                recalls.update(recall, bs)

                precision = precision_score(y.cpu(), output.argmax(-1), average='macro')
                precisions.update(precision, bs)

                try:
                    rocauc = roc_auc_score(y.cpu(), np.amax(output, axis=1))
                    roaucs.update(rocauc, bs)
                except:
                    rocauc = 0.0 # and we will not record this.

                writer.add_scalar("loss/valid", loss.item(), global_step=cur_step)
                writer.add_scalar("f1/valid", f1, global_step=cur_step)
                writer.add_scalar("precision/valid", precision, global_step=cur_step)
                writer.add_scalar("recall/valid", recall, global_step=cur_step)
                writer.add_scalar("rocauc/valid", rocauc, global_step=cur_step)

                if step % config.log_frequency == 0 or step == len(valid_loader) - 1:
                    logger.info(
                        "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                        "F1 ({f1s.avg:.1%}) Precision({precisions.avg:.1%}) Recall({recalls.avg:.1%}) RocAuc({rocaucs.avg:.1%}) ".format( # 
                            epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                            f1s=f1s, precisions=precisions, recalls=recalls, rocaucs=rocaucs))  #

                
        # writer.add_scalar("loss/test", losses.avg, global_step=cur_step)
        # writer.add_scalar("f1/test", f1s.avg, global_step=cur_step)
        # writer.add_scalar("precisio/test", precisions.avg, global_step=cur_step)
        # writer.add_scalar("recall/test", recalls.avg, global_step=cur_step)
        # writer.add_scalar("rocauc/test", rocaucs.avg, global_step=cur_step)
        # # writer.add_scalar("acc1/test", top1.avg, global_step=cur_step)
        # # writer.add_scalar("acc5/test", top5.avg, global_step=cur_step)

        # logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))
        # logger.info("Valid: [{:3d}/{}] Final (avg) F1@1 {:.4%} Precisions {:.4%} Recalls {:.4%} RocAucs {:.4%}".format(
        #     epoch + 1, config.epochs, f1s.avg, precisions.avg, recalls.avg, rocaucs.avg))
        logger.info("Valid: [{:3d}/{}] Final: F1 ({f1s.avg:.1%}) Precision({precisions.avg:.1%}) "
                    "Recall({recalls.avg:.1%}) RocAuc({rocaucs.avg:.1%}) ".format(  #
                            epoch + 1, config.epochs, 
                            f1s=f1s, precisions=precisions, recalls=recalls, rocaucs=rocaucs))  #


        return rocaucs.avg, f1s.avg

    def log_final_validation_results(data_loader, model, device):
        
        f1s = AverageMeter("f1")
        precisions = AverageMeter("precision")
        recalls = AverageMeter("recall")

        model.eval()

        with torch.no_grad():
            s = 1
            for step, (X, y) in enumerate(data_loader):
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                bs = X.size(0)

                logits = model(X)
                loss = criterion(logits, y)

                # f1_val = utils.f1(logits, y)
                # losses.update(loss.item(), bs)
                # f1.update(accuracy["f1"], bs)

                output = logits.cpu().detach().numpy()
                f1 = f1_score(y.cpu(), output.argmax(-1), average='macro')
                f1s.update(f1, bs)

                recall = recall_score(y.cpu(), output.argmax(-1), average='macro')
                recalls.update(recall, bs)

                precision = precision_score(y.cpu(), output.argmax(-1), average='macro')
                precisions.update(precision, bs)

                writer.add_scalar("f1/valid", f1, global_step=s)
                writer.add_scalar("precision/valid", precision, global_step=s)
                writer.add_scalar("recall/valid", recall, global_step=s)

            logger.info(
                "F1 ({f1s.avg:.1%}) Precision({precisions.avg:.1%}) Recall({recalls.avg:.1%})".format(
                    f1s=f1s, precisions=precisions, recalls=recalls))
            s = s + 1

    CLASSES_SET_ = args.classes_set

    n_classes = len(CLASSES_SET_)
    valid_channels = args.valid_channels
    n_channels = len(valid_channels)

    folds_partions_dict, labels = data_gen.train_val_split_strat_kfolds(folds_n=5, classes_list=CLASSES_SET_)

    # Augment train data with paths to folders with augmented melanomas
    aug_c43_folder = '/mnt/data/interim/_melanoma_artificial2/C43'
    augmented_train_c43_paths = []
    for __f in os.listdir(aug_c43_folder):
        augmented_train_c43_paths.append(os.path.join(aug_c43_folder, __f))

    labels_aug_c43 = {}
    for path_ in augmented_train_c43_paths:
        labels_aug_c43[path_] = 0
    labels.update(labels_aug_c43)

    from random import randrange, sample
    def __random_insert_seq(lst, seq):
        insert_locations = sample(range(len(lst) + len(seq)), len(seq))
        inserts = dict(zip(insert_locations, seq))
        input = iter(lst)
        lst[:] = [inserts[pos] if pos in inserts else next(input)
            for pos in range(len(lst) + len(seq))]

    for __k in folds_partions_dict.keys():
        target_lst = folds_partions_dict[__k]['train']
        __random_insert_seq(target_lst, augmented_train_c43_paths)
        folds_partions_dict[__k]['train'] = target_lst
    # END OF Augment train data with paths to folders with augmented melanomas

    for fold_n, partition in folds_partions_dict.items():
        
        #####################
        target_folder_name = args.target_folder_name  #'128x128_nonnorm_5lrs_16ch_6chimg_7sept_set2_2classes_v2.2_registered'
        target_folder_name = target_folder_name + '_fold-{}'.format(fold_n)
        os.makedirs(target_folder_name, exist_ok=True)

        logger = logging.getLogger('nni-darts')  # root logger - Good to get it only once.
        for hdlr in logger.handlers[:]:  # remove the existing file handlers
            if isinstance(hdlr, logging.FileHandler): #fixed two typos here
                logger.removeHandler(hdlr)

        logger.setLevel(logging.DEBUG)
        hdlr = logging.FileHandler(target_folder_name+'/model1.log.txt')
        formatter = logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr) 
        ####################3

        logger.info(args)
    
        logger.info('CLASSES_SET_:')
        for CLZ in CLASSES_SET_:
            logger.info(CLZ.__str__())
        logger.info('---------------')

        logger.info('fold data details:')
        logger.info('fold_n - {}'.format(fold_n))
        logger.info('partition->{}'.format(partition))

        device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
        writer = SummaryWriter()

        model = CNN(input_size=320, 
                    in_channels=n_channels, 
                    channels=11,
                    n_classes=n_classes, 
                    n_layers=args.layers)
        # model = nn.DataParallel(model)
        
        logger.info('loading model checkgpoint {}'.format(args.arc_checkpoint))

        apply_fixed_architecture(model, args.arc_checkpoint)
        criterion = nn.CrossEntropyLoss()

        model.to(device)
        criterion.to(device)

        optimizer = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1E-6)
        #####################

        train_loader, valid_loader = prepare_train_val_generators(partition, labels)

        best_rocauc = 0.0
        best_f1 = 0.0
        for epoch in range(args.epochs):
            drop_prob = args.drop_path_prob * epoch / args.epochs
            model.drop_path_prob(drop_prob)

            # training
            train(args, train_loader, model, optimizer, criterion, epoch)

            # validation
            cur_step = (epoch + 1) * len(train_loader)
            roc_auc_val, f1_val = validate(args, valid_loader, model, criterion, epoch, cur_step)
            best_rocauc = max(best_rocauc, roc_auc_val)

            if f1_val > best_f1:
                save_model( path_to_folder=target_folder_name, 
                            file_name='epoch_{}_model1.pt'.format(epoch),
                            model=model)
            best_f1 = max(best_f1, f1_val)

            lr_scheduler.step()


        logger.info('1 -=-=-=-=-=-=-=-=-=-=-=-=-')
        logger.info('FINALS Valid:')
        log_final_validation_results(valid_loader, model, device)
        logger.info('FINALS Train:')
        log_final_validation_results(valid_loader, model, device)
        logger.info('2 -=-=-=-=-=-=-=-=-=-=-=-=-')
        logger.info('Valid: \n' + str(calc_conf_mat(valid_loader, n_classes, device, model)))
        logger.info('Train: \n' + str(calc_conf_mat(train_loader, n_classes, device, model)))
        logger.info('3 -=-=-=-=-=-=-=-=-=-=-=-=-')
        logger.info("Final best RocAuc@1 = {:.4%}".format(best_rocauc))
        save_model( path_to_folder=target_folder_name, 
                    file_name='final_model1.pt',
                    model=model)

if __name__ == "__main__":
    # args = get_arguments(sys.argv[1:])

    # import logreset


    '''
    1 vs. 2; 
    1 vs. 2 vs. 3;
    1 vs. 2 vs. 3 vs. 4

    1) Melanoma: C43 + D03+ D03.9
    2) Pigmented benign: D22 + L81 + L81.2 + L81.4 + Q82.5 
    3) Keratin lesions: D86.3 + L21 + L57 + L57.0 + L82 + L85 + L85.1 + L85.5 + L85.8 + Q80
    4) Nonmelanoma skin cancer: C44 + C46 + D09
    '''
    ROOT_PATHS = [  Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/checkyourskin'),
                Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/loc_ALL_WHITE'),
                Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE_Split_Channels_MAN_CORRECTED/loc_old_colored'),
                Path('/mnt/data/interim/_melanoma_20200728_2__checkyourskin_c43_NoDuplicates_14Nov2020')]

    CLS1 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43', 'd03', 'd03.9'], class_name='Melanoma_like_lesions', int_label=0)
    CLS2 = ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22', 'l81.2','l81.4', 'q82.5'], class_name='Pigmented_benign', int_label=1)   #'d81', 
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

    print('len(CLASSES_SET_9)=', len(CLASSES_SET_9))

    layers_list = [1,2,3]

    cuda_id_ = 0
    do_visualize = True
    # CHANNELS = ['r-r', 'ir-r', 'g-g', 'uv-0-r', 'uv-0-g', 'white-g', 'white-r']
    # CHANNELS = ['r-r', 'ir-r', 'g-g', 'uv-0-r', 'uv-0-g', 'white-g']
    CHANNELS = ['r-r', 'ir-r', 'g-g', 'uv-0-g']
    channels_to_zero_out = []  #['ir-r']

    epochs = 350  #750
    batch_size = 4

    runs_args = []

    for L in layers_list:
        runs_args.append(
            SimpleNamespace(layers=L, batch_size=batch_size, log_frequency=10, epochs=epochs, channels=11, unrolled=False, \
                            drop_path_prob=0.2, grad_clip=5.0,\
                            arc_checkpoint='/mnt/models/multi_search_14Dec2020_128x128_4ch_12ch/search_{}lr_4ch_5classes_12innerCh/epoch_49.json'.format(L), \
                            cuda_id=cuda_id_, \
                            target_folder_name='/mnt/models/darts_retrained/4ch_128x128_12inchannel_no_metainfo_registered_5Fold_newC43_augC43_v1_21Dec2020_32classes/XV2_SGD_orig_02DropChannel_{}lrs_15Dec_newC43NoDup_ClassSet4_ManCorected_registered_350epochs'.format(L), \
                            valid_channels=CHANNELS, \
                            channels_to_zero_out=channels_to_zero_out, \
                            channel_drop_prob=0.2, \
                            classes_set=CLASSES_SET_9)
                            )

    for args in runs_args:
        
        # logreset.reset_logging()
        main(args)