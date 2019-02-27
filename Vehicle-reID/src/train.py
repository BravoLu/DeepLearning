from __future__ import print_function, absolute_import 
import argparse
import os
import numpy as np
import sys
import os.path as osp
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import time
import dataset
import transforms as T
from Preprocessor import *
import models
from trainer import ResNetTrainer,RAJNetTrainer
#from evaluators import Evaluator
from utils.serialization import load_checkpoint, save_checkpoint
from loss import FocalLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

def adjust_lr(epoch, args, optimizer):
    step_size = 60
    lr = args.lr * ( 0.1 ** (epoch // step_size))
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)

def main(args):
    end = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    train_set, val_set, query_set, gallery_set, train_loader, val_loader, test_loader = dataset.get_data(args.dataset, args.batch_size, args.workers, args.test_size)
    
    model = models.create(args.arch)
    start_epoch = best_top1 = 0

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Statr epoch {} best_top1 {:.1%}"
                .format(start_epoch, best_top1))
    
    model = nn.DataParallel(model).cuda() 
    
    # Criterion
    #criterion = nn.CrossEntropyLoss().cuda()
    criterion = FocalLoss().cuda()
    
    if hasattr(model.module,'base'):
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() 
                        if id(p) not in base_param_ids]

        params_groups = [
            {'params': model.module.base.parameters(), 'lr_mult':0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        params_groups = model.parameters()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay,
                                   nesterov=True)
    # Trainer
    #trainer = ResNetTrainer(model, criterion)
    trainer = RAJNetTrainer(model, criterion)

    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch,args, optimizer)
        trainer.train(epoch, train_loader, optimizer)
        
        if epoch < args.start_save:
            continue

        #top1 = evaluator.evaluate(val_loader, val_set, val_set)
        #is_best = top1 > best_top1
        #best_top1 = max(top1, best_top1)
        #best_top1 = 0
        #save_checkpoint({
        #    'state_dict': model.module.state_dict(),
        #    'epoch': epoch + 1,
        #    'best_top1' : best_top1,
        #}, is_best=True, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        #print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
        #        format(epoch, top1, best_top1, ' *' if is_best else ''))

        #print('#epoch {}'.format(epoch))
    
    #Test 
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, query_set, gallery_set)
    print("time consumed:{:.2%} s".format(time.time() - end))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vehicle re-ID")
    #mobilenetv1, mobilenetv2, ResNet18
    parser.add_argument('-a', '--arch', type=str, default='RAJNet18')
    parser.add_argument('-d', '--dataset', type=str, default='vehicleid_v1.0')
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-j', '--workers',  type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--start_save', type=int, default=0)
    parser.add_argument('--test_size', type=int,default=1600)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'logs/RAJNet18'))
    main(parser.parse_args())
          


        
