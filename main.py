import numpy as np
import os
import random
import wandb

import torch
import argparse
import timm
import logging
import yaml

from stats import dataset_stats
from train import fit
from timm import create_model
from datasets import create_dataloader
from log import setup_default_logging

_logger = logging.getLogger('train')

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


def run(cfg):
    # make save directory
    savedir = os.path.join(cfg['RESULT']['savedir'], cfg['DATASET']['dataname'], cfg['EXP_NAME'])
    os.makedirs(savedir, exist_ok=True)

    setup_default_logging(log_path=os.path.join(savedir,'log.txt'))
    torch_seed(cfg['SEED'])

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # build Model
    model = create_model(cfg['MODEL'], num_classes=cfg['DATASET']['num_classes'], pretrained=True)
    model.to(device)
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model.parameters()])))

    # load dataset
    trainset, testset = __import__('datasets').__dict__[f"load_{cfg['DATASET']['dataname'].lower()}"](
        datadir            = cfg['DATASET']['datadir'], 
        img_size           = cfg['DATASET']['img_size'],
        mean               = cfg['DATASET'].get('mean', None), 
        std                = cfg['DATASET'].get('std', None),
        normalize          = cfg['DATASET']['normalize']
    )
    
    # load dataloader
    trainloader = create_dataloader(dataset=trainset, batch_size=cfg['TRAINING']['batch_size'], shuffle=True)
    testloader = create_dataloader(dataset=testset, batch_size=cfg['TRAINING']['test_batch_size'], shuffle=False)

    # set training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[cfg['OPTIMIZER']['opt_name']](model.parameters(), lr=cfg['OPTIMIZER']['lr'])

    # scheduler
    if cfg['TRAINING']['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['TRAINING']['epochs'])
    else:
        scheduler = None

    # initialize wandb
    wandb.init(name=cfg['EXP_NAME'], project='Normalization Test', config=cfg)

    # fitting model
    fit(model        = model, 
        trainloader  = trainloader, 
        testloader   = testloader, 
        criterion    = criterion, 
        optimizer    = optimizer, 
        scheduler    = scheduler,
        epochs       = cfg['TRAINING']['epochs'], 
        savedir      = savedir,
        log_interval = cfg['TRAINING']['log_interval'],
        device       = device)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Normalization Test')
    parser.add_argument('--default_setting', type=str, default=None, help='exp config file')    
    parser.add_argument('--modelname', type=str, default='resnet18', help='model name')
    parser.add_argument('--dataname', type=str, default='CIFAR10', help='data name')
    parser.add_argument('--normalize', type=str, default='finetune', choices=['finetune','pretrain','instance','minmax'], help='normlization setting')

    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.default_setting,'r'), Loader=yaml.FullLoader)
    
    d_stats = dataset_stats[args.dataname.lower()]
    
    cfg['MODEL'] = args.modelname
    cfg['DATASET']['num_classes'] = d_stats['num_classes']
    cfg['DATASET']['dataname'] = args.dataname
    cfg['DATASET']['normalize'] = args.normalize
    cfg['DATASET']['img_size'] = d_stats['img_size']
    
    if args.normalize == 'finetune':
        cfg['DATASET']['mean'] = d_stats['mean']
        cfg['DATASET']['std'] = d_stats['std']
    elif args.normalize == 'pretrain':
        cfg['DATASET']['mean'] = dataset_stats['imagenet']['mean']
        cfg['DATASET']['std'] = dataset_stats['imagenet']['std']
        
    cfg['EXP_NAME'] = f"{args.modelname}-{args.normalize}"

    run(cfg)