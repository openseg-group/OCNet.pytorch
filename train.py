##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: speedinghzl02
## Modified by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
from dataset import get_segmentation_dataset
from network import get_segmentation_model
from config import Parameters
import random
import timeit
import logging
import pdb
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.criterion import CriterionCrossEntropy,  CriterionDSN, CriterionOhemDSN, CriterionOhemDSN_single
from utils.parallel import DataParallelModel, DataParallelCriterion


start = timeit.default_timer()

args = Parameters().parse()

# file_log = open(args.log_file, "w")
# sys.stdout = sys.stderr = file_log

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))
   
   
def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def main():
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    writer = SummaryWriter(args.snapshot_dir)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    cudnn.enabled = True

    deeplab = get_segmentation_model("_".join([args.network, args.method]), num_classes=args.num_classes)

    saved_state_dict = torch.load(args.restore_from)
    new_params = deeplab.state_dict().copy()

    if 'wide' in args.network:
        saved_state_dict = saved_state_dict['state_dict']
        if 'vistas' in args.method:
            saved_state_dict = saved_state_dict['body']
            for i in saved_state_dict:
                new_params[i] = saved_state_dict[i]
        else:     
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not 'classifier' in i_parts: 
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
    elif 'mobilenet' in args.network:
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not (i_parts[0]=='features' and i_parts[1]=='18') and not i_parts[0]=='classifier':
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i] 
    else:
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[0]=='fc' and not  i_parts[0]=='last_linear' and not  i_parts[0]=='classifier':
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i] 

    if args.start_iters > 0:
        deeplab.load_state_dict(saved_state_dict)
    else:
        deeplab.load_state_dict(new_params)

    model = DataParallelModel(deeplab)
    # model = nn.DataParallel(deeplab)
    model.train()     
    model.float()
    model.cuda()    

    criterion = CriterionCrossEntropy()
    if "dsn" in args.method:
        if args.ohem:
            if args.ohem_single:
                print('use ohem only for the second prediction map.')
                criterion = CriterionOhemDSN_single(thres=args.ohem_thres, min_kept=args.ohem_keep, dsn_weight=float(args.dsn_weight))
            else:
                criterion = CriterionOhemDSN(thres=args.ohem_thres, min_kept=args.ohem_keep, dsn_weight=float(args.dsn_weight), use_weight=True)
        else:
            criterion = CriterionDSN(dsn_weight=float(args.dsn_weight), use_weight=True)


    criterion = DataParallelCriterion(criterion)
    criterion.cuda()
    cudnn.benchmark = True


    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(get_segmentation_dataset(args.dataset, root=args.data_dir, list_path=args.data_list,
                    max_iters=args.num_steps*args.batch_size, crop_size=input_size, 
                    scale=args.random_scale, mirror=args.random_mirror, network=args.network), 
                    batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    optimizer = optim.SGD([{'params': filter(lambda p: p.requires_grad, deeplab.parameters()), 'lr': args.learning_rate }], 
                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)


    optimizer.zero_grad()

    for i_iter, batch in enumerate(trainloader):
        sys.stdout.flush()
        i_iter += args.start_iters
        images, labels, _, _ = batch
        images = Variable(images.cuda())
        labels = Variable(labels.long().cuda())
        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter)
        if args.fix_lr:
            lr = args.learning_rate
        print('learning_rate: {}'.format(lr))

        if 'gt' in args.method:
            preds = model(images, labels)
        else:
            preds = model(images)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        if i_iter % 100 == 0:
            writer.add_scalar('learning_rate', lr, i_iter)
            writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)
        print('iter = {} of {} completed, loss = {}'.format(i_iter, args.num_steps, loss.data.cpu().numpy()))

        if i_iter >= args.num_steps-1:
            print('save model ...')
            torch.save(deeplab.state_dict(),osp.join(args.snapshot_dir, 'CS_scenes_'+str(args.num_steps)+'.pth'))
            break

        if i_iter % args.save_pred_every == 0:
            print('taking snapshot ...')
            torch.save(deeplab.state_dict(),osp.join(args.snapshot_dir, 'CS_scenes_'+str(i_iter)+'.pth'))     

    end = timeit.default_timer()
    print(end-start,'seconds')

if __name__ == '__main__':
    main()
