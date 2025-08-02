import os
import sys
import time
import json
import copy
import torch
import random
import pickle
import argparse
import itertools
import datetime
import pytz

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from collections import namedtuple
from typing import Type, Any, Callable, Union, List, Optional

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.nn.functional as F
import torch.optim as optim
from torch import nn, Tensor
from torch.distributions import Categorical
from torch.utils.data import Dataset, random_split
import torch.utils.data as data
import torchvision.models as models

from ..models.new_models import FullyConnectedNetwork
from ..models.model_dict import get_model_from_name
from ..utils.core import (
    get_model_infos, time_string, convert_secs2time
)
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

def cifar_100_train_eval_loop( args, logger, epoch, optimizer, scheduler, network, xloader, criterion, batch_size, mode='eval' ):
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
        train_data, valid_data = data.random_split(train_data, [len(train_data)-len(train_data)//5,  len(train_data)//5])
        test_data, valid_data = data.random_split(valid_data, [len(valid_data)-len(valid_data)//2,  len(valid_data)//2])
    
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

    # creating multiple STUDENTs
    model_name = [ "ResNet10_xxxs","ResNet10_xxs","ResNet10_xs","ResNet10_s","ResNet10_m" ]
    k= len(model_name)

    # set mode here
    Mode= 1    
    image_encoder=models.resnet18(pretrained=True).cuda()
    image_encoder = nn.Sequential(*list(image_encoder.children())[:-1])
    
    base_model= [i for i in range(k)]
    network= [i for i in range(k)]
    best_state_dict= [i for i in range(k)]
    
    for i in range(k):
        base_model[i] = get_model_from_name( model_config, model_name[i] )
        logger.log("Student {} + {}:".format(i, model_name[i]) )
    
        ce_ptrained_path = "./ce_results/CE_with_seed-{}_cycles-1_{}-{}"\
                            "model_best.pth.tar".format(args.rand_seed,
                                                        args.dataset,
                                                        model_name[i])
    
        # Loading multiple students from pretrained student  
        logger.log("using pretrained student model from {}".format(ce_ptrained_path))
                        
        if args.pretrained_student: # load CE-pretrained student
            assert Path().exists(), "Cannot find the initialization file : {:}".format(ce_ptrained_path)
        base_checkpoint = torch.load(ce_ptrained_path)
        base_model[i].load_state_dict(base_checkpoint["base_state_dict"])
            
    for i in range(k):
        base_model[i] = base_model[i].cuda()
        network[i] = base_model[i] 
        best_state_dict[i] = copy.deepcopy(base_model[i].state_dict())
        
    # testing pretrained student
    for i in range(k):
        test_loss, test_acc1, test_acc5 = evaluate_model(network[i], test_loader, criterion, args.eval_batch_size)
        logger.log(
        "***{:s}*** before training [Student(CE)] {} Test loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f}, error@1 = {:.2f}, error@5 = {:.2f}".format(
            time_string(),i,
            test_loss,
            test_acc1,
            test_acc5,
            100 - test_acc1,
            100 - test_acc5,
        )
        )
    
    optimizer_s=[i for i in range(k)]
    scheduler_s=[i for i in range(k)]
    
    for i in range(k):
        optimizer_s[i] = torch.optim.SGD(base_model[i].parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
        scheduler_s[i] = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_s[i], args.epochs//args.sched_cycles)
        logger.log("Scheduling LR update to student no {}, {} time at {}-epoch intervals".format(k,args.sched_cycles, args.epochs//args.sched_cycles))

    # TEACHER
    Teacher_model = get_model_from_name(model_config, args.teacher)
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

    flop, param = get_model_infos(base_model[0], xshape)
    args.base_flops = flop 
    logger.log("model information : {:}".format(base_model[0].get_message()))
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
        if Mode==1:
            # Image is fed directly to ResNet
            Arguments_meta = namedtuple("Configure", ('class_num','dataset')  )
            md_dict_meta = { 'class_num' : 2*k, 'dataset' : args.dataset }
            model_config_meta = Arguments_meta(**md_dict_meta)
            meta_net= get_model_from_name( model_config_meta, "ResNet10_s" )
            meta_net= meta_net.cuda()
        if Mode==2:
            # Encoded img to ResNet
            meta_net = ResNet32MetaNet().cuda()
        if Mode==3:
            # Encoded img to FC Layer
            meta_net = FullyConnectedNetwork(512,[128,256,517,1024,128], 2*k ).cuda()
        if Mode==4:
            # Logits to FC
            meta_net = FullyConnectedNetwork(100*k,[128,256,512,1024,2048,1000,500,200,100,50,25,10], 2*k ).cuda()
        if Mode==5:
            # Logits + Encoded img to FC
            meta_net = FullyConnectedNetwork(512+100*k,[128,256,512,1024,2048,1000,500,200,100,50,25,10], 2*k ).cuda()
        if Mode==6:
            # Logits + Encoded img to ResNet
            meta_net = ResNet32MetaNet().cuda()
            
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)

    best_acc, best_epoch = 0.0, 0
    val_losses = []
    alphas_collection = [[] for i in range(k)]
    betas_collection = [[] for i in range(k)]
      
    Temp = args.temperature
    log_file_name = get_model_prefix( args )
    
    loss= [i for i in range(k)] 
    for epoch in range(args.epochs):
        logger.log("\nStarted EPOCH:{}".format(epoch))
        mode='train'
        logger.log('Training epoch', epoch)
        
        losses = [ AverageMeter('Loss', ':.4e') for i in range(k)]
        top1 = [ AverageMeter('Acc@1', ':6.2f') for i in range(k)]
        top5 = [ AverageMeter('Acc@5', ':6.2f') for i in range(k)]
        progress= [ i for i in range(k)]
        for i in range(k):
            base_model[i].train()
            progress[i] = ProgressMeter(
                    logger,
                    len(train_loader),
                    [losses[i], top1[i], top5[i]],
                    prefix="[{}] E: [{}]".format(mode.upper(), epoch))
        
        for iteration, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.cuda()
            targets = labels.cuda(non_blocking=True)
            # train metanet
            if (iteration + 1) % args.meta_interval == 0:
                pseudo_net= [i for i in range(k)]
                features= [i for i in range(k)]
                pseudo_outputs= [i for i in range(k)]  
                 
                # make a descent in a COPY OF THE STUDENT (in train data), metanet net will do a move on this move for metaloss
                for i in range(k):
                    pseudo_net[i] = get_model_from_name( model_config, model_name[i] )
                    pseudo_net[i] = pseudo_net[i].cuda()
                    pseudo_net[i].load_state_dict(network[i].state_dict()) # base_model == network
                    pseudo_net[i].train()
                    
                    features[i], pseudo_outputs[i], _= pseudo_net[i](inputs)
                
                meta_net.train()    
                
                with torch.no_grad():
                    _, teacher_outputs, _ = network_t(inputs)
                
                pseudo_loss_vector_CE= [ i for i in range(k)]
                for i in range(k):
                    pseudo_loss_vector_CE[i]= criterion_indiv(pseudo_outputs[i], targets) # [B]
                
                if args.meta_type == 'meta_lite':
                    pseudo_hyperparams = meta_net(features)    
                else:
                    image_features = image_encoder(inputs)
                    tensors = logits.copy()
                    if Mode==1:
                        pseudo_hyperparams,_,_ = meta_net(inputs)
                    if Mode==2:
                        with torch.no_grad():
                            image_features= torch.cat((image_features, image_features), dim=1)
                            image_features=image_features.squeeze().view(image_features.shape[0], 32, 32).unsqueeze(1).expand(-1, 3, -1, -1)
                            pseudo_hyperparams = meta_net(image_features)
                    if Mode==3:
                        with torch.no_grad():
                            pseudo_hyperparams = meta_net(image_features.squeeze())
                    if Mode==4:
                        with torch.no_grad():
                            encoded_img= tensors[0]
                            for i in range(k-1):
                                encoded_img= torch.cat([ encoded_img,tensors[i+1]], dim=1)
                            pseudo_hyperparams = meta_net(encoded_img)
                    if Mode==5:
                        with torch.no_grad():
                            encoded_img= image_features.squeeze()
                            for i in range(k):
                                encoded_img= torch.cat([ encoded_img,tensors[i]], dim=1)
                            pseudo_hyperparams = meta_net(encoded_img)
                    if Mode==6:
                        with torch.no_grad():
                            encoded_img= image_features.squeeze()
                            for i in range(k):
                                encoded_img= torch.cat([ encoded_img,tensors[i]], dim=1)
                            zeros_tensor = torch.zeros(encoded_img.size()[0], 112).cuda()
                            encoded_img= torch.cat([ encoded_img,zeros_tensor], dim=1)
                            encoded_img=encoded_img.unsqueeze(1).unsqueeze(2)
                            encoded_img=encoded_img.reshape(encoded_img.shape[0],32, 32).unsqueeze(1).expand(-1, 3, -1, -1)
                            pseudo_hyperparams = meta_net(encoded_img)
                            
                alpha = [i for i in range(k)]
                beta = [i for i in range(k)]
                
                for i in range(k):
                    alpha[i] = pseudo_hyperparams[:,2*i]
                    beta[i] = pseudo_hyperparams[:,2*i+1]
                    
                Temp = args.temperature
                pseudo_loss_vector_KD= [ i for i in range(k)]
                for i in range(k):
                    pseudo_loss_vector_KD[i]= nn.KLDivLoss(reduction='none')(F.log_softmax(pseudo_outputs[i] / Temp, dim=1),
                                                                             F.softmax(teacher_outputs / Temp, dim=1))
                
                # Loss Update
                loss_CE=[i for i in range(k)]
                loss_KD=[i for i in range(k)]
                
                pseudo_loss= [i for i in range(k)]
                for i in range(k):
                    loss_CE[i] = torch.mean( alpha[i]*pseudo_loss_vector_CE[i] )
                    loss_KD[i] = (Temp**2)* torch.mean( beta[i] * torch.sum(pseudo_loss_vector_KD[i],dim=1))
                    pseudo_loss[i] = loss_CE[i] + loss_KD[i] 
                pseudo_grads=[i for i in range(k)]
                
                for i in range(k):
                    pseudo_grads[i] = torch.autograd.grad(pseudo_loss[i], pseudo_net[i].parameters(), create_graph=True)

                    # using the current student's LR to train pseudo network
                    base_model_lr = optimizer_s[i].param_groups[0]['lr']
                    pseudo_optimizer = MetaSGD(pseudo_net[i], pseudo_net[i].parameters(), lr=base_model_lr)
                    pseudo_optimizer.load_state_dict(optimizer_s[i].state_dict())
                    pseudo_optimizer.meta_step(pseudo_grads[i])

                # To save space on CPU
                del pseudo_grads

                # metanet descent
                # cycle through the metadata used for validation
                try:
                    valid_inputs, valid_labels = next(meta_dataloader_iter)
                except StopIteration:
                    meta_dataloader_iter = iter(meta_loader)
                    valid_inputs, valid_labels = next(meta_dataloader_iter)

                valid_inputs, valid_labels = valid_inputs.cuda(), valid_labels.cuda()
                
                # Calculate MetaNet Loss
                meta_loss = 0
                meta_output=[]
                for i in range(k):
                    _,meta_outputs,_ = pseudo_net[i](valid_inputs) # apply the stepped pseudo net on the validation data
                    meta_output.append(meta_outputs)
                    meta_loss += torch.mean(criterion_indiv(meta_outputs, valid_labels.long()))
                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()
            
            features= [i for i in range(k)]
            logits= [i for i in range(k)]
            loss_vector= [i for i in range(k)]
            
            for i in range(k):
                features[i], logits[i], _ = network[i](inputs)
            with torch.no_grad():
                _,teacher_outputs , _ = network_t(inputs)

                if args.meta_type == 'meta_lite':
                    hyperparams = meta_net(features)    
                else:
                    image_features = image_encoder(inputs)
                    tensors = logits.copy()
                    if Mode==1:
                        hyperparams,_,_ = meta_net(inputs)
                    if Mode==2:
                        with torch.no_grad():
                            image_features= torch.cat((image_features, image_features), dim=1)
                            image_features=image_features.squeeze().view(image_features.shape[0], 32, 32).unsqueeze(1).expand(-1, 3, -1, -1)
                            hyperparams = meta_net(image_features)
                    if Mode==3:
                        with torch.no_grad():
                            hyperparams = meta_net(image_features.squeeze())
                    if Mode==4:
                        with torch.no_grad():
                            encoded_img= tensors[0]
                            for i in range(k-1):
                                encoded_img= torch.cat([ encoded_img,tensors[i+1]], dim=1)
                            hyperparams = meta_net(encoded_img)
                    if Mode==5:
                        with torch.no_grad():
                            encoded_img= image_features.squeeze()
                            for i in range(k):
                                encoded_img= torch.cat([ encoded_img,tensors[i]], dim=1)
                            hyperparams = meta_net(encoded_img)
                    if Mode==6:
                        with torch.no_grad():
                            encoded_img= image_features.squeeze()
                            for i in range(k):
                                encoded_img= torch.cat([ encoded_img,tensors[i]], dim=1)
                            zeros_tensor = torch.zeros(encoded_img.size()[0], 112).cuda()
                            encoded_img= torch.cat([ encoded_img,zeros_tensor], dim=1)
                            encoded_img=encoded_img.unsqueeze(1).unsqueeze(2)
                            encoded_img=encoded_img.reshape(encoded_img.shape[0],32, 32).unsqueeze(1).expand(-1, 3, -1, -1)
                            hyperparams = meta_net(encoded_img)

            # Insialize the loss weights 
            alpha = [i for i in range(k)]
            beta = [i for i in range(k)] 
            
            alphas = [i for i in range(k)]
            betas = [i for i in range(k)]

            for j in range(k):
                alpha[j] = hyperparams[:,2*j]
                beta[j] = hyperparams[:,2*j +1]
                
                alphas[j] = alpha[j].detach()
                betas[j] = beta[j].detach() 
                
                if iteration == 0:
                    alphas[j] = alpha[j].cpu()
                    betas[j] = beta[j].cpu()
                else:
                    alphas[j] = torch.cat((alphas[j].cpu() ,alpha[j].cpu()), dim =0)
                    betas[j] = torch.cat((betas[j].cpu() ,beta[j].cpu()), dim =0)

            for i in range(k):
                optimizer_s[i].zero_grad()
                loss_vector = criterion_indiv(logits[i], targets)
                loss_vector_KD= nn.KLDivLoss(reduction='none')(F.log_softmax(logits[i] / Temp, dim=1),\
                                                               F.softmax(teacher_outputs / Temp, dim=1))
                loss_CE= torch.mean( alpha[i]*loss_vector )
                loss_KD = (Temp**2)* torch.mean( beta[i] * torch.sum(loss_vector_KD,dim=1))
                
                loss= loss_CE+loss_KD
                prec1, prec5 = obtain_accuracy(logits[i].data, targets.data, topk=(1, 5))
                
                loss.backward()
                optimizer_s[i].step()        
                
                losses[i].update(loss.item(), inputs.size(0))
                top1[i].update(prec1.item(), inputs.size(0))
                top5[i].update(prec5.item(), inputs.size(0))
            for i in range(k):
                if (iteration % args.print_freq == 0) or (iteration == len(train_loader)-1):
                    progress[i].display(iteration)
        
        if epoch%199==0 or epoch==args.epochs-1:
            for i in range(k):
                alphas_collection[i].append(alphas[i])
                betas_collection[i].append(betas[i])
        
        for i in range(k):
            scheduler_s[i].step(epoch)

        best_acc= [0 for i in range(k)]
        for i in range(k):
            val_loss, val_acc1, val_acc5 = cifar_100_train_eval_loop( args, logger, epoch, optimizer_s[i], scheduler_s[i], network[i], valid_loader, criterion, args.eval_batch_size, mode='eval' )
            is_best = False 
            if val_acc1 > best_acc[i]:
                best_acc[i] = val_acc1
                is_best = True
                best_state_dict[i] = copy.deepcopy( network[i].state_dict() )
                best_epoch = epoch+1
            save_checkpoint({
                    'epoch': epoch + 1,
                    'student': i,
                    'base_state_dict': base_model[i].state_dict(),
                    'best_acc': best_acc[i],
                    'meta_state_dict': meta_net.state_dict(),
                    'scheduler_s' : scheduler_s[i].state_dict(),
                    'optimizer_s' : optimizer_s[i].state_dict(),
                }, is_best, prefix=log_file_name)
            val_losses.append(val_loss)
            logger.log('std {} Valid eval after epoch: loss:{:.4f}\tlatest_acc:{:.2f}\tLR:{:.4f} -- best valacc {:.2f}'.format(i,val_loss,
                                                                                                                               val_acc1,
                                                                                                                               get_mlr(scheduler_s[i]), 
                                                                                                                               best_acc[i]))

    for i in range(k):
        network[i].load_state_dict(best_state_dict[i])
        test_loss, test_acc1, test_acc5 = evaluate_model(network[i], test_loader, criterion, args.eval_batch_size)
        logger.log(
            "\n***{:s}*** [Post-train] [Student {}] Test loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f}, error@1 = {:.2f}, error@5 = {:.2f}".format(
                time_string(),i,
                test_loss,
                test_acc1,
                test_acc5,
                100 - test_acc1,
                100 - test_acc5,
            )
        )
        logger.log("Result is from best val model {} of epoch:{}".format(i,best_epoch))
    
    plots_dir = os.path.join(args.save_dir, args.file_name)
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    # Save the loss weights
    for i in range(k):
        with open(os.path.join(plots_dir, 'alpha_{}_{}.pkl'.format(model_name[i],i)), 'wb') as f:
            pickle.dump(alphas_collection, f)
            logger.log("Saved intermediate weights to {}".format(os.path.join(plots_dir, 'alpha_dump.pkl')))
        with open(os.path.join(plots_dir, 'beta_{}_{}.pkl'.format(model_name[i],i)), 'wb') as f:
            pickle.dump(betas_collection, f)
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
    parser.add_argument("--batch_size", type=int, default=400, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=400, help="Batch size for testing.")
    parser.add_argument('--epochs', type=int, default=100,help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=10e-4   ,help='learning rate for a single GPU')
    parser.add_argument('--momentum', type=float, default=0.9,help='SGD momentum')
    parser.add_argument("--pretrained_student", type=int, default=1, help="should I use CE-pretrained student?")
    parser.add_argument('--wd', type=float, default=0.0005,  help='weight decay')
    parser.add_argument('--label', type=str, default="",  help='give some label you want appended to log fil')
    parser.add_argument('--temperature', type=int, default=4,  help='temperature for KD')
    parser.add_argument('--sched_cycles', type=int, default=1,  help='How many times cosine cycles for scheduler')
    parser.add_argument('--file_name', type=str, default="",  help='file_name')
    
    ################################## MC-Distil specific arguments #####
    parser.add_argument('--inst_based', type=bool, default=True)
    parser.add_argument('--meta_interval', type=int, default=20)
    parser.add_argument('--mcd_weight', type=float, default=0.5)
    parser.add_argument('--meta_weight_decay', type=float, default=1e-4)
    parser.add_argument('--input_size', type=int, default=32)
    parser.add_argument('--meta_lr', type=float, default=1e-3)
    parser.add_argument('--unsup_adapt', type=bool, default=False)
    parser.add_argument('--meta_type', type=str, default='resnet') # resnet, meta_lite, instance
    #####################################################################
    
    args = parser.parse_args()

    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 10)
    if (args.file_name is None or args.file_name == ""):
        if args.pretrained_student==1:
            args.file_name = "meta_net_{}_meta_seed-{}_metalr-{}_T-{}_{}_{}-cycles_{}_{}".format(args.dataset,
                                                            args.rand_seed, 
                                                            args.meta_lr,
                                                            args.temperature,
                                                            args.epochs,
                                                            args.sched_cycles,
                                                            args.lr,
                                                            args.momentum)
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



