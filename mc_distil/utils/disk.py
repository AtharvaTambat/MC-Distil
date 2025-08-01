"""
DiSK: Distillation Support Kit
Utility functions for knowledge distillation experiments.
"""

import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
from .logging import AverageMeter

def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""  
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_mlr(scheduler):
    """Get the current learning rate from scheduler"""
    if hasattr(scheduler, 'get_last_lr'):
        return scheduler.get_last_lr()[0]
    elif hasattr(scheduler, 'optimizer'):
        return scheduler.optimizer.param_groups[0]['lr']
    else:
        # Fallback for simple optimizer
        return scheduler.param_groups[0]['lr']

def save_checkpoint(state, filename, logger):
    """Save model checkpoint"""
    if osp.isfile(filename):
        if hasattr(logger, "log"):
            logger.log(
                "Find {:} exist, delete is at first before saving".format(filename)
            )
        os.remove(filename)
    torch.save(state, filename)
    assert osp.isfile(
        filename
    ), "save filename : {:} failed, which is not found.".format(filename)

def evaluate_model(network, xloader, criterion, batch_size):
    """Evaluate model on given dataset"""
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    network.eval()
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(xloader):
            inputs = inputs.cuda()
            targets = targets.cuda(non_blocking=True)
            
            # Handle different model output formats
            output = network(inputs)
            if isinstance(output, tuple):
                # If model returns tuple (like _, logits, _), get the logits
                if len(output) == 3:
                    _, logits, _ = output
                else:
                    logits = output[1] if len(output) > 1 else output[0]
            else:
                logits = output
            
            loss = criterion(logits, targets)
            prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
            
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
    
    return losses.avg, top1.avg, top5.avg