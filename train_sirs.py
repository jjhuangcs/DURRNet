import os
from os.path import join

import torch.backends.cudnn as cudnn

import data.sirs_dataset as datasets
import util.util as util
from data.image_folder import read_fns
from engine import Engine
from options.net_options.train_options import TrainOptions
from tools import mutils
import random
import torch
import numpy as np


opt = TrainOptions().parse()
print(opt)
cudnn.benchmark = True

opt.display_freq = 10

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 1
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 9999
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

# modify the following code to
datadir = os.path.join(os.path.expanduser('~'), 'input the directory of the training dataset')

datadir_syn = join(datadir, 'train/VOCdevkit/VOC2012/JPEGImages')
datadir_real = join(datadir, 'train/real')

train_dataset = datasets.CEILDataset(
    datadir_syn, read_fns('data/VOC2012_224_train_jpg.txt'), size=opt.max_dataset_size, enable_transforms=True,
    low_sigma=opt.low_sigma, high_sigma=opt.high_sigma,
    low_gamma=opt.low_gamma, high_gamma=opt.high_gamma)

train_dataset_real = datasets.CEILTrainDataset(datadir_real, read_fns('data/real_train.txt'),
                                              enable_transforms=True, if_align=opt.if_align)

train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_real], [0.7, 0.3])


train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataset_real = datasets.CEILTestDataset(join(datadir, f'test420/real20_{opt.real20_size}'),
                                             fns=read_fns('data/real_test.txt'),
                                             enable_transforms=False,
                                             if_align=opt.if_align)
eval_dataset_solidobject = datasets.CEILTestDataset(join(datadir, 'test/SIR2/SolidObjectDataset'),
                                                    if_align=opt.if_align)
eval_dataset_postcard = datasets.CEILTestDataset(join(datadir, 'test/SIR2/PostcardDataset'), if_align=opt.if_align)
eval_dataset_wild = datasets.CEILTestDataset(join(datadir, 'test/SIR2/WildSceneDataset'), if_align=opt.if_align)

eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_solidobject = datasets.DataLoader(
    eval_dataset_solidobject, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)
eval_dataloader_postcard = datasets.DataLoader(
    eval_dataset_postcard, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_wild = datasets.DataLoader(
    eval_dataset_wild, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

"""Main Loop"""
engine = Engine(opt)
result_dir = os.path.join(f'./checkpoints/{opt.name}/results',
                          mutils.get_formatted_time())


def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)


if opt.resume or opt.debug_eval:
    save_dir = os.path.join(result_dir, '%03d' % engine.epoch)
    os.makedirs(save_dir, exist_ok=True)
    engine.eval(eval_dataloader_real, dataset_name='testdata_real20', savedir=save_dir, suffix='real20')
    engine.eval(eval_dataloader_solidobject, dataset_name='testdata_solidobject', savedir=save_dir,
                suffix='solidobject')
    engine.eval(eval_dataloader_postcard, dataset_name='testdata_postcard', savedir=save_dir, suffix='postcard')
    engine.eval(eval_dataloader_wild, dataset_name='testdata_wild', savedir=save_dir, suffix='wild')

# define training strategy
engine.model.opt.lambda_gan = 0
set_learning_rate(opt.lr)

decay_rate = 0.5
while engine.epoch < opt.nEpochs:
    if opt.fixed_lr == 0:
        if engine.epoch >= opt.nEpochs * 0.2:
            engine.model.opt.lambda_gan = 0.0001  # gan loss is added
        if engine.epoch >= opt.nEpochs * 0.4:
            set_learning_rate(opt.lr * decay_rate**1)#0.5
        if engine.epoch >= opt.nEpochs * 0.6:
            set_learning_rate(opt.lr * decay_rate**2)#0.2
        if engine.epoch >= opt.nEpochs * 0.8:
            set_learning_rate(opt.lr * decay_rate**3)#0.1
        if engine.epoch >= opt.nEpochs:
            set_learning_rate(opt.lr * decay_rate**4)#0.05
    else:
        set_learning_rate(opt.fixed_lr)

    engine.train(train_dataloader_fusion)

    if engine.epoch % 1 == 0:
        save_dir = os.path.join(result_dir, '%03d' % engine.epoch)
        os.makedirs(save_dir, exist_ok=True)
        engine.eval(eval_dataloader_real, dataset_name='testdata_real20', savedir=save_dir, suffix='real20')
        engine.eval(eval_dataloader_solidobject, dataset_name='testdata_solidobject', savedir=save_dir,
                    suffix='solidobject')
        engine.eval(eval_dataloader_postcard, dataset_name='testdata_postcard', savedir=save_dir, suffix='postcard')
        engine.eval(eval_dataloader_wild, dataset_name='testdata_wild', savedir=save_dir, suffix='wild')
