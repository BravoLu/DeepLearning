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
from evaluators import Evaluator
import dataset
import transforms as T
from Preprocessor import *
import models
from trainer import *
#from evaluators import Evaluator
from utils.serialization import load_checkpoint, save_checkpoint
from loss import TripletLoss,FocalLoss
import loss

#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

def adjust_lr(epoch, args,optimizer):
    step_size = 60
    lr = args.lr * ( 0.1 ** (epoch // step_size))
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)



def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    train_set, val_set, query_set ,gallery_set, train_loader, val_loader, test_loader = dataset.get_data(args.dataset, args.batch_size, args.workers, args.test_size, 128, 128)
    
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
    if args.loss in loss.LOSS_SET:
        criterion = loss.LOSS_SET[args.loss]
    else:
        raise KeyError("Unknown loss:",name) 
    
    if hasattr(model.module,'base'):
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() 
                        if id(p) not in base_param_ids]

        params_groups = [
            {'params': model.module.base.parameters(), 'lr_mult':0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        params_groups = model.parameters()

    
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay,
                                   nesterov=True)

    trainer = ADFLTrainer(model, criterion)
    evaluator = Evaluator(model)
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch,args, optimizer)
        trainer.train(epoch, train_loader, optimizer)
        
        if epoch < args.start_save:
            continue

        _, top1 = evaluator.evaluate(test_loader, query_set, gallery_set)
        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        #best_top1 = 0
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1' : best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, args.ckpt))

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
                format(epoch, top1, best_top1, ' *' if is_best else ''))

        print('#epoch {}'.format(epoch))
    
    #Test 
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, query_set, gallery_set)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="V35")
    #mobilenetv1, mobilenetv2, ResNet18
    parser.add_argument('-l','--loss', type=str, default='CrossEntropy')
    parser.add_argument('-a', '--arch', type=str, default='ADFLNet')
    parser.add_argument('-d', '--dataset', type=str, default='vehicleid_v1.0')
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-j', '--workers',  type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--print-freq', type=int, default=1)
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--start_save', type=int, default=0)
    parser.add_argument('--test_size', type=int,default=1600)
    parser.add_argument('--margin', type=float, default=0.5, help="margin of the triplet loss, default: 0.5")
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'logs/ADFLNet_CrossEntropyLoss'))
    parser.add_argument('--ckpt', type=str, default='2019-2-21.pth.tar')
    main(parser.parse_args())
          


        
