import numpy as np
import os
import random
import wandb

import torch
import argparse
import timm
import logging
import yaml

from train import fit
from timm import create_model
from datasets import create_dataset, create_dataloader
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
    model = create_model(cfg['MODEL']['modelname'], num_classes=cfg['MODEL']['num_classes'], pretrained=True)
    model.to(device)
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model.parameters()])))

    # load dataset
    trainset, testset = create_dataset(
        datadir  = cfg['DATASET']['datadir'], 
        dataname = cfg['DATASET']['dataname'], 
        mean     = cfg['DATASET'].get('mean', model.default_cfg['mean']), 
        std      = cfg['DATASET'].get('std', model.default_cfg['std'])
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
    parser.add_argument('--yaml_config', type=str, default=None, help='exp config file')    

    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.yaml_config,'r'), Loader=yaml.FullLoader)
    

    run(cfg)