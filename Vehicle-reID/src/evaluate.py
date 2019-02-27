import sys
import os.path as osp
import torch
import argparse
import os
import numpy as np
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import dataset
import transforms as T
from Preprocessor import *
#from trainer import Trainer
from utils.serialization import load_checkpoint, save_checkpoint
import models
from evaluators import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    
    _, _, query_set, gallery_set, _, _, test_loader = dataset.get_data(args.dataset, args.batch_size, args.workers, args.test_size, 128, 128)
    #for i,_ in enumerate(test_loader):
        #print(i)
    model = models.create(args.arch)
    model = nn.DataParallel(model).cuda()
 
    evaluator = Evaluator(model)

    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'ADFLNet_CrossEntropyLoss/model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, query_set, gallery_set )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluation")
    #mobilenetv1, mobilenetv2, ResNet18
    parser.add_argument('-a', '--arch', type=str, default='ADFLNet')
    parser.add_argument('-d', '--dataset', type=str, default='vehicleid_v1.0')
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-j', '--workers',  type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--print-freq', type=int, default=1)
    parser.add_argument('--resume', type=str, default='logs/checkpoint.pth.tar', metavar='PATH')
    parser.add_argument('--start_save', type=int, default=0)
    parser.add_argument('--test_size', type=int,default=800)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'logs'))
    main(parser.parse_args())
          

