import os
import sys
import logging
from argparse import ArgumentParser
import torch
import datasets

from putils import get_parameters
from model_melanoma_v1 import SearchMelanomaV1MobileNet
# from nni.nas.pytorch.proxylessnas import ProxylessNasTrainer

from retrain import Retrain


from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass


sys.path.append('/mnt')
sys.path.append('/mnt/skin_cancer_nas')
sys.path.append('/mnt/skin_cancer_nas/data/torch_generator')
sys.path.append('/mnt/skin_cancer_nas/skin_cancer_nas/nas/proxylessnas_nni_melanoma')
from skin_cancer_nas.data.torch_generator import generator as data_gen
from skin_cancer_nas.data.torch_generator import base_classes
from skin_cancer_nas.data.torch_generator.config import *
from base_classes import Dataset
from torchvision import transforms

from skin_cancer_nas.nas.proxylessnas_nni_melanoma.trainer_melanoma import ProxylessNasTrainerMelanoma

from config import CLASSES_SET
# CLASSES_SET = [ ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43'], class_name='C43', int_label=1), 
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22'], class_name='D22', int_label=0)]
n_classes = len(CLASSES_SET)



logger = logging.getLogger('nni_proxylessnas')

if __name__ == "__main__":
    parser = ArgumentParser("proxylessnas")
    # configurations of the model
    # parser.add_argument("--n_cell_stages", default='4,4,4,4,4,1', type=str)
    # parser.add_argument("--stride_stages", default='2,2,2,1,2,1', type=str)
    parser.add_argument("--n_cell_stages", default='4,4,1', type=str)
    parser.add_argument("--stride_stages", default='2,2,1', type=str)
    # parser.add_argument("--width_stages", default='24,40,80,96,192,320', type=str)
    parser.add_argument("--width_stages", default='24,40,96', type=str)
    parser.add_argument("--bn_momentum", default=0.1, type=float)
    parser.add_argument("--bn_eps", default=1e-3, type=float)
    parser.add_argument("--dropout_rate", default=0, type=float)
    parser.add_argument("--no_decay_keys", default='bn', type=str, choices=[None, 'bn', 'bn#bias'])
    # configurations of imagenet dataset
    parser.add_argument("--data_path", default='/data/imagenet/', type=str)
    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--test_batch_size", default=100, type=int)
    parser.add_argument("--n_worker", default=4, type=int)
    parser.add_argument("--resize_scale", default=0.08, type=float)
    parser.add_argument("--distort_color", default='normal', type=str, choices=['normal', 'strong', 'None'])
    # configurations for training mode
    parser.add_argument("--train_mode", default='search', type=str, choices=['search', 'retrain'])
    # configurations for search
    parser.add_argument("--checkpoint_path", default='/mnt/models/proxylessnas/128x128_c43_d22_newdata_v2_aug1/search_mobile_net.pt', type=str)
    parser.add_argument("--arch_path", default='/mnt/models/proxylessnas/128x128_c43_d22_newdata_v2_aug1/arch_path.pt', type=str)
    parser.add_argument("--no-warmup", dest='warmup', action='store_false')
    # configurations for retrain
    parser.add_argument("--exported_arch_path", default=None, type=str)

    args = parser.parse_args()
    if args.train_mode == 'retrain' and args.exported_arch_path is None:
        logger.error('When --train_mode is retrain, --exported_arch_path must be specified.')
        sys.exit(-1)

    model = SearchMelanomaV1MobileNet(width_stages=[int(i) for i in args.width_stages.split(',')],
                            n_cell_stages=[int(i) for i in args.n_cell_stages.split(',')],
                            stride_stages=[int(i) for i in args.stride_stages.split(',')],
                            n_classes=n_classes,
                            dropout_rate=args.dropout_rate,
                            bn_param=(args.bn_momentum, args.bn_eps))
    logger.info('SearchMelanomaV1MobileNet model create done')
    model.init_model()
    logger.info('SearchMelanomaV1MobileNet model init done')

    # move network to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda') # :0
    else:
        device = torch.device('cpu')

    data_device = device  #torch.device('cpu')

    logger.info('Creating data provider...')
    # This is original ImageNet data provider
    data_provider = datasets.C43_D22_v1_DataProvider(   save_path=args.data_path,
                                                        train_batch_size=args.train_batch_size,
                                                        test_batch_size=args.test_batch_size,
                                                        valid_size=None,
                                                        n_worker=args.n_worker,
                                                        resize_scale=args.resize_scale,
                                                        distort_color=args.distort_color,
                                                        device=data_device,
                                                        pin_memory=False)

    logger.info('Creating data provider done')

    if args.no_decay_keys:
        keys = args.no_decay_keys
        momentum, nesterov = 0.9, True
        optimizer = torch.optim.SGD([
            {'params': get_parameters(model, keys, mode='exclude'), 'weight_decay': 4e-5},
            {'params': get_parameters(model, keys, mode='include'), 'weight_decay': 0},
        ], lr=0.05, momentum=momentum, nesterov=nesterov)
    else:
        optimizer = torch.optim.SGD(get_parameters(model), lr=0.05, momentum=momentum, nesterov=nesterov, weight_decay=4e-5)

    if args.train_mode == 'search':
        # this is architecture search
        logger.info('Creating ProxylessNasTrainerMelanoma...')
        trainer = ProxylessNasTrainerMelanoma(model,
                                      model_optim=optimizer,
                                      train_loader=data_provider.train,
                                      valid_loader=data_provider.valid,
                                      device=device,
                                      warmup=args.warmup,
                                      ckpt_path=args.checkpoint_path,
                                      arch_path=args.arch_path)

        logger.info('Start to train with ProxylessNasTrainerMelanoma...')
        trainer.train()
        logger.info('Training done')
        trainer.export(args.arch_path)
        logger.info('Best architecture exported in %s', args.arch_path)
    elif args.train_mode == 'retrain':
        # this is retrain
        from nni.nas.pytorch.fixed import apply_fixed_architecture
        assert os.path.isfile(args.exported_arch_path), \
            "exported_arch_path {} should be a file.".format(args.exported_arch_path)
        apply_fixed_architecture(model, args.exported_arch_path)
        trainer = Retrain(model, optimizer, device, data_provider, n_epochs=300)
        trainer.run()