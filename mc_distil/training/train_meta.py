import os
import sys
import time
import json
import random
import argparse
import pickle
import itertools
import datetime
import pytz
import copy
from pathlib import Path
from collections import namedtuple
from typing import Type, Any, Callable, Union, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn, Tensor
from torch.distributions import Categorical
from torch.utils.data import Dataset, random_split
import torch.utils.data as data

from ..models.model_dict import get_model_from_name
from ..utils.core import get_model_infos, time_string, convert_secs2time
from ..utils.logging import AverageMeter, ProgressMeter
from ..utils.initialization import prepare_logger, prepare_seed
from ..utils.disk import obtain_accuracy, get_mlr, save_checkpoint, evaluate_model
from ..data.get_dataset_with_transform import get_datasets
from .meta import *
from ..models.base import *


def m__get_prefix(args):
    prefix = args.file_name + '_' + args.dataset + '-' + args.model_name
    return prefix

def get_model_prefix(args):
    prefix = os.path.join(args.save_dir, m__get_prefix(args))
    return prefix

def cifar_100_train_eval_loop(args, logger, epoch, optimizer, scheduler, network, xloader, criterion, batch_size, mode='eval'):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    if mode == 'eval': 
        network.eval()
    else:
        network.train()

    progress = ProgressMeter(
            logger,
            len(xloader),
            [losses, top1, top5],
            prefix="[{}] E: [{}]".format(mode.upper(), epoch))

    for i, (inputs, targets) in enumerate(xloader):
        if mode == 'train':
            optimizer.zero_grad()

        inputs = inputs.cuda()
        targets = targets.cuda(non_blocking=True)
        _, logits, _ = network(inputs)

        loss = torch.mean(criterion(logits, targets))
        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if (i % args.print_freq == 0) or (i == len(xloader)-1):
                progress.display(i)

    return losses.avg, top1.avg, top5.avg

def main(args):
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.workers)
    logger = prepare_logger(args)
    prepare_seed(args.rand_seed)

    criterion_indiv = nn.CrossEntropyLoss(reduction= 'none')
    criterion = nn.CrossEntropyLoss()
    
    dataset=args.dataset
    if dataset=="cifar100" or dataset=="cifar10":
        train_data, test_data, xshape, class_num = get_datasets(args.dataset, args.data_path, args.cutout_length)
        train_data, valid_data = data.random_split(train_data, [len(train_data)-len(train_data)//10, len(train_data)//10])
    else:
        train_data, test_data, xshape, class_num = get_datasets(args.dataset, args.data_path, args.cutout_length)
        train_data, valid_data = data.random_split(train_data, [len(train_data)-len(train_data)//5, len(train_data)//5])
        test_data, valid_data = data.random_split(valid_data, [len(valid_data)-len(valid_data)//2, len(valid_data)//2])
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    meta_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader( # same data used for training metanet
        valid_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    meta_dataloader_iter = iter(train_loader)
    args.class_num = class_num
    
    logger.log(args.__str__())
    logger.log("Train:{}\t, Valid:{}\t, Test:{}\n".format(len(train_data),
                                                          len(valid_data),
                                                          len(test_data)))
    logger.log("-" * 50)
    Arguments = namedtuple("Configure", ('class_num','dataset')  )
    md_dict = { 'class_num' : class_num, 'dataset' : args.dataset }
    model_config = Arguments(**md_dict)

    # STUDENT
    base_model = get_model_from_name( model_config, args.model_name )
    logger.log("Student :" + args.model_name)

    base_model = base_model.cuda()
    network = base_model
    best_state_dict = copy.deepcopy( base_model.state_dict() )
    ce_ptrained_path = "./ce_results/CE_with_seed-{}_cycles-1_{}-{}"\
                        "model_best.pth.tar".format(args.rand_seed,
                                                    args.dataset,
                                                    args.model_name)
                        
    if args.pretrained_student: # load CE-pretrained student
        assert Path().exists(), "Cannot find the initialization file : {:}".format(ce_ptrained_path)
        logger.log("using pretrained student model from {}".format(ce_ptrained_path))
        base_checkpoint = torch.load(ce_ptrained_path)
        base_model.load_state_dict(base_checkpoint["base_state_dict"])

    # testing pretrained student
    test_loss, test_acc1, test_acc5 = evaluate_model( network, test_loader, criterion, args.eval_batch_size )
    logger.log(
        "***{:s}*** before training [Student(CE)] Test loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f}, error@1 = {:.2f}, error@5 = {:.2f}".format(
            time_string(),
            test_loss,
            test_acc1,
            test_acc5,
            100 - test_acc1,
            100 - test_acc5,
        )
    )
    
    optimizer_s = torch.optim.SGD(base_model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler_s = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_s, args.epochs//args.sched_cycles)
    logger.log("Scheduling LR update to student {} time at {}-epoch intervals".format(args.sched_cycles, 
                                                                                      args.epochs//args.sched_cycles))

    # TEACHER
    Teacher_model = get_model_from_name( model_config, args.teacher )
    teach_PATH = "./ce_results/CE_with_seed-{}_cycles-1_{}-{}"\
                    "model_best.pth.tar".format(args.rand_seed,
                                                args.dataset,
                                                args.teacher)
                    
    teach_checkpoint = torch.load(teach_PATH)
    Teacher_model.load_state_dict(teach_checkpoint['base_state_dict'])
    Teacher_model = Teacher_model.cuda()
    network_t = Teacher_model
    network_t.eval()
    logger.log("Teacher loaded....")
    
    # testing teacher
    test_loss, test_acc1, test_acc5 = evaluate_model( network_t, test_loader, nn.CrossEntropyLoss(), args.eval_batch_size )
    logger.log(
        "***{:s}*** [Teacher] Test loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f}, error@1 = {:.2f}, error@5 = {:.2f}".format(
            time_string(),
            test_loss,
            test_acc1,
            test_acc5,
            100 - test_acc1,
            100 - test_acc5,
        )
    )

    flop, param = get_model_infos(base_model, xshape)
    args.base_flops = flop 
    logger.log("model information : {:}".format(base_model.get_message()))
    logger.log("-" * 50)
    logger.log("[Student]Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format(
            param, flop, flop / 1e3))

    # METANET
    if args.meta_type == 'meta_lite':
        meta_net = InstanceMetaNetLite(num_layers=1).cuda()
    elif args.meta_type == 'instance':
        logger.log("Using Instance metanet....")
        meta_net = InstanceMetaNet(input_size=args.input_size).cuda()
    else:
        logger.log("Using ResNet32 metanet....")
        Arguments_meta = namedtuple("Configure", ('class_num','dataset')  )
        md_dict_meta = { 'class_num' : 2, 'dataset' : args.dataset }
        model_config_meta = Arguments_meta(**md_dict_meta)
        meta_net= get_model_from_name( model_config_meta, "ResNet10_s" )
        meta_net= meta_net.cuda()
    
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)

    best_acc, best_epoch = 0.0, 0
    val_losses = []
    log_alphas_collection = []
    log_betas_collection = []
      
    Temp = args.temperature
    log_file_name = get_model_prefix(args)
       
    for epoch in range(args.epochs):
        logger.log("\nStarted EPOCH:{}".format(epoch))
        mode='train'
        logger.log('Training epoch', epoch)
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        
        base_model.train()
        progress = ProgressMeter(
                logger,
                len(train_loader),
                [losses, top1, top5],
                prefix="[{}] E: [{}]".format(mode.upper(), epoch))
        
        for iteration, (inputs, labels) in enumerate(train_loader): 
            inputs = inputs.cuda()
            targets = labels.cuda(non_blocking=True)
            # train metanet
            if (iteration + 1) % args.meta_interval == 0:
                # make a descent in a COPY OF THE STUDENT (in train data), metanet net will do a move on this move for metaloss
                pseudo_net = get_model_from_name( model_config, args.model_name )
                pseudo_net = pseudo_net.cuda()
                pseudo_net.load_state_dict(network.state_dict()) # base_model == network
                pseudo_net.train()
                meta_net.train()    
                
                features, pseudo_outputs, _= pseudo_net(inputs)
                with torch.no_grad():
                    _, teacher_outputs, _ = network_t(inputs)

                pseudo_loss_vector_CE = criterion_indiv(pseudo_outputs, targets) # [B]
                
                if args.meta_type == 'meta_lite':
                    pseudo_hyperparams = meta_net(features)    
                else:
                    pseudo_hyperparams,_,_ = meta_net(inputs)
                alpha = pseudo_hyperparams[:,0]
                beta = pseudo_hyperparams[:,1]
                
                Temp = args.temperature
                pseudo_loss_vector_KD = nn.KLDivLoss(reduction='none')(         # [B x n]
                                    F.log_softmax(pseudo_outputs / Temp, dim=1),
                                    F.softmax(teacher_outputs / Temp, dim=1))
                
                
                # Loss Update
                loss_CE = torch.mean( alpha * pseudo_loss_vector_CE )
                loss_KD = (Temp**2)* torch.mean( beta * torch.sum(pseudo_loss_vector_KD,dim=1))
                
                pseudo_loss = loss_CE + loss_KD
                
                pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)

                # using the current student's LR to train pseudo
                base_model_lr = optimizer_s.param_groups[0]['lr']
                pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=base_model_lr)
                pseudo_optimizer.load_state_dict(optimizer_s.state_dict())
                pseudo_optimizer.meta_step(pseudo_grads)

                # To save space
                del pseudo_grads

                # metanet descent
                # cycle through the metadata used for validation
                try:
                    valid_inputs, valid_labels = next(meta_dataloader_iter)
                except StopIteration:
                    meta_dataloader_iter = iter(meta_loader)
                    valid_inputs, valid_labels = next(meta_dataloader_iter)

                valid_inputs, valid_labels = valid_inputs.cuda(), valid_labels.cuda()
                _,meta_outputs,_ = pseudo_net(valid_inputs) # apply the stepped pseudo net on the validation data

                meta_loss = torch.mean(criterion_indiv(meta_outputs, valid_labels.long()))
                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()
            
            optimizer_s.zero_grad()

            features, logits, _ = network(inputs)
            loss_vector = criterion_indiv(logits, targets)   

            with torch.no_grad():
                _,teacher_outputs , _ = network_t(inputs)
            
            if args.meta_type == 'meta_lite':
                hyperparams = meta_net(features)    
            else:
                hyperparams,_,_ = meta_net(inputs)
                
            alpha = hyperparams[:,0]
            beta = hyperparams[:,1]
            alpha = alpha.detach()
            beta = beta.detach()

            if iteration == 0:
                alphas = alpha.cpu()
                betas = beta.cpu()
            else:
                alphas = torch.cat((alphas,alpha.cpu()), dim =0)
                betas = torch.cat((betas,beta.cpu()), dim =0)

            pseudo_loss_vector_KD = nn.KLDivLoss(reduction='none')(F.log_softmax(logits / Temp, dim=1),\
                                                               F.softmax(teacher_outputs / Temp, dim=1))
            
            # Loss Update
            loss_CE = torch.mean( alpha * loss_vector)
            loss_KD = (Temp**2)* torch.mean(beta * torch.sum(pseudo_loss_vector_KD, dim=1))

            loss = loss_CE + loss_KD
            prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))

            loss.backward()
            optimizer_s.step()

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            
            if (iteration % args.print_freq == 0) or (iteration == len(train_loader)-1):
                progress.display(iteration)

        if epoch%199==0 or epoch==args.epochs-1:
            log_alphas_collection.append(alphas)
            log_betas_collection.append(betas)

        scheduler_s.step(epoch)
        val_loss, val_acc1, val_acc5 = cifar_100_train_eval_loop( args, logger, epoch, optimizer_s, scheduler_s, network, valid_loader, criterion, args.eval_batch_size, mode='eval' )
        is_best = False 
        if val_acc1 > best_acc:
            best_acc = val_acc1
            is_best = True
            best_state_dict = copy.deepcopy( network.state_dict() )
            best_epoch = epoch+1
        save_checkpoint({
                'epoch': epoch + 1,
                'base_state_dict': base_model.state_dict(),
                'best_acc': best_acc,
                'meta_state_dict': meta_net.state_dict(),
                'scheduler_s' : scheduler_s.state_dict(),
                'optimizer_s' : optimizer_s.state_dict(),
            }, is_best, prefix=log_file_name)
        val_losses.append(val_loss)
        logger.log('Valid eval after epoch: loss:{:.4f}\tlatest_acc:{:.2f}\tLR:{:.4f} -- best valacc {:.2f}'.format( val_loss,
                                                                                                                        val_acc1,
                                                                                                                        get_mlr(scheduler_s), 
                                                                                                                        best_acc))

    network.load_state_dict( best_state_dict )
    test_loss, test_acc1, test_acc5 = evaluate_model( network, test_loader, criterion, args.eval_batch_size )
    logger.log(
        "\n***{:s}*** [Post-train] [Student] Test loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f}, error@1 = {:.2f}, error@5 = {:.2f}".format(
            time_string(),
            test_loss,
            test_acc1,
            test_acc5,
            100 - test_acc1,
            100 - test_acc5,
        )
    )
    logger.log("Result is from best val model of epoch:{}".format(best_epoch))
    
    plots_dir = os.path.join(args.save_dir, args.file_name)
    
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    # Save the loss weights given by metanet
    with open(os.path.join(plots_dir, 'alpha_dump.pkl'), 'wb') as f:
        pickle.dump(log_alphas_collection, f)
        logger.log("Saved intermediate weights to {}".format(os.path.join(plots_dir, 'alpha_dump.pkl')))
    with open(os.path.join(plots_dir, 'beta_dump.pkl'), 'wb') as f:
        pickle.dump(log_betas_collection, f)
        logger.log("Saved intermediate weights to {}".format(os.path.join(plots_dir, 'beta_dump.pkl')))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a classification model on typical image classification datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset", type=str, default='cifar10', help="The dataset name.")
    parser.add_argument("--data_path", type=str, default='./data/', help="The dataset name.")
    parser.add_argument("--model_name", type=str, default='ResNet32TwoPFiveM-NAS', help="The path to the model configuration")
    parser.add_argument("--teacher", type=str, default='ResNet10_l', help="teacher model name")
    parser.add_argument("--cutout_length", type=int, default=16, help="The cutout length, negative means not use.")
    parser.add_argument("--print_freq", type=int, default=100, help="print frequency (default: 200)")
    parser.add_argument("--print_freq_eval", type=int, default=100, help="print frequency (default: 200)")
    parser.add_argument("--save_dir", type=str, help="Folder to save checkpoints and log.", default='./logs/')
    parser.add_argument("--workers", type=int, default=8, help="number of data loading workers (default: 8)")
    parser.add_argument("--rand_seed", type=int, help="base model seed")
    parser.add_argument("--global_rand_seed", type=int, default=-1, help="global model seed")
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=200, help="Batch size for testing.")
    parser.add_argument('--epochs', type=int, default=100,help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.05,help='learning rate for a single GPU')
    parser.add_argument('--momentum', type=float, default=0.9,help='SGD momentum')
    parser.add_argument("--pretrained_student", type=int, default=1, help="should I use CE-pretrained student?")
    parser.add_argument('--wd', type=float, default=0.0005,  help='weight decay')
    parser.add_argument('--label', type=str, default="",  help='give some label you want appended to log fil')
    parser.add_argument('--temperature', type=int, default=4,  help='temperature for KD')
    parser.add_argument('--sched_cycles', type=int, default=1,  help='How many times cosine cycles for scheduler')
    parser.add_argument('--file_name', type=str, default="",  help='file_name')
    
    ###################### MC-Distil specific arguments #################
    parser.add_argument('--inst_based', type=bool, default=True)
    parser.add_argument('--meta_interval', type=int, default=20)
    parser.add_argument('--mcd_weight', type=float, default=1.0)
    parser.add_argument('--meta_weight_decay', type=float, default=1e-3)
    parser.add_argument('--input_size', type=int, default=32)
    parser.add_argument('--meta_lr', type=float, default=1e-5)
    parser.add_argument('--unsup_adapt', type=bool, default=False)
    parser.add_argument('--meta_type', type=str, default='resnet') # or meta_lite, resnet
    #####################################################################
    
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 10)
    if (args.file_name is None or args.file_name == ""):
        if args.pretrained_student==1:
            args.file_name = "{}_{}_meta_seed-{}_metalr-{}_T-{}_{}_{}-cycles".format(args.model_name,args.dataset,
                                                            args.rand_seed, 
                                                            args.meta_lr,
                                                            args.temperature,
                                                            args.epochs,
                                                            args.sched_cycles)
        else:
            args.file_name = "meta_no_PT_seed-{}_metalr-{}_T-{}_{}_{}-cycles".format(
                                                            args.rand_seed, 
                                                            args.meta_lr,
                                                            args.temperature,
                                                            args.epochs,
                                                            args.sched_cycles)
    args.file_name += '_'+args.meta_type
    assert args.save_dir is not None, "save-path argument can not be None"

    main(args)



