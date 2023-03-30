# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:
import argparse
import copy
import os
import os.path as osp
import warnings
import random
import numpy as np
import torch
import torch.distributed as dist
import time

# 1.14s

import sys
sys.path.append('../..')
sys.path.append('../mmcv')

from UDL.AutoDL import build_model, getDataSession, ModelDispatcher
from UDL.Basis.auxiliary import init_random_seed, set_random_seed
from mmcv.utils.logging import print_log, create_logger
# 1.5s
from mmcv.runner import init_dist, find_latest_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)


# 10s
# from mmdet.datasets import (build_dataloader, build_dataset,
#                             replace_ImageToTensor)

def trainer(cfg, logger,
            distributed=False,
            meta=None):

    # TODO: 对于多个model进行任务的封装的时候，放进构建器里，而不是这里？ 似乎会增加构建代价
    # TODO: 构建
    model, criterion, optimizer, scheduler = build_model(cfg.arch, cfg.task, cfg)


    if hasattr(model, 'init_weights'):
        model.init_weights()

    ############################################################
    # 不适合多任务
    ############################################################
    # datasets = [build_dataset(cfg.data.train)]
    # if len(cfg.workflow) == 2:
    #     val_dataset = copy.deepcopy(cfg.data.val)
    #     val_dataset.pipeline = cfg.data.train.pipeline
    #     datasets.append(build_dataset(val_dataset))
    # model.CLASSES = datasets[0].CLASSES

    # prepare data loaders
    # datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]

    # runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
    #     'type']
    # data_loaders = [
    #     build_dataloader(
    #         ds,
    #         cfg.samples_per_gpu,
    #         cfg.workers_per_gpu,
    #         # `num_gpus` will be ignored if distributed
    #         num_gpus=1,
    #         dist=distributed,
    #         seed=args.seed,
    #         runner_type=runner_type,
    #         persistent_workers=cfg.data.get('persistent_workers', False))
    #     for ds in datasets
    # ]

    sess = getDataSession(cfg)
    cfg.valid_or_test = False
    if cfg.eval:
        cfg.workflow = [('test', 1)]
    if not any('train' in mode for mode, _ in cfg.workflow):
        cfg.eval = True
    if not any('valid' in mode for mode, _ in cfg.workflow):
        cfg.valid_or_test = True

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        # TODO: ugly access classmethod
        '''
        相应的改变了model.train() [runner:54-60], model.eval() [runner:153-158], model.backward() [OptimizerHook: 57-87],  save/load checkpoint [checkpoint: 283-310, runner.load_checkpoint]这四个部分
        '''
        if not hasattr(model, 'train'):  # 任务分配器是否注册为模型，注册了就会有'train'
            if isinstance(model.model, dict):  # 实际运行的模型可以有多个，通过字典区分
                for name, m in model.model.items():  # model不是模型, model.model是字典
                    model.model[name] = MMDataParallel(m, device_ids=cfg.gpu_ids)
            else:
                model.model = MMDataParallel(model.model, device_ids=cfg.gpu_ids)
        else:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    # 改到 build_model里，一次性设置，方便查找
    if cfg.get('optimizer', None) is not None:
        optimizer = build_optimizer(model, cfg.optimizer)

    # 兼容argparser和配置文件的
    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.epochs  # argparser
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'epochs' in cfg and 'max_iters' not in cfg.runner:
            cfg.runner['max_epochs'] = cfg.epochs
            # assert cfg.epochs == cfg.runner['max_epochs'], print(cfg.epochs, cfg.runner['max_epochs'])

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
            opt_cfg={'print_freq': cfg.print_freq,
                     'accumulated_step': cfg.accumulated_step,
                     'clip_max_norm': cfg.clip_max_norm,
                     'dataset': cfg.dataset,
                     'img_range': cfg.img_range,
                     'metrics': cfg.metrics,
                     'save_fmt': cfg.save_fmt,
                     'mode': cfg.mode,
                     'eval': cfg.eval,
                     'val_mode': cfg.valid_or_test, # 在base_runner的resume里用于设置测试最大轮数来评估训练好的模型
                     'save_dir': cfg.work_dir + "/results"}))

    # an ugly workaround to make .log and .log.json filenames the same
    # runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.get('optimizer_config', None)

    ############################################################
    # register training hooks
    ############################################################
    if cfg.get('config', None) is not None:
        '''
        optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
        optimizer_config = dict(grad_clip=None)
        lr_config = dict(policy='step', step=[100, 150])
        checkpoint_config = dict(interval=1)
        log_config = dict(
            interval=100,
            hooks=[
                dict(type='TextLoggerHook'),
                # dict(type='TensorboardLoggerHook')
            ])
        '''
        runner.register_training_hooks(
            cfg.lr_config,
            optimizer_config,
            cfg.checkpoint_config,
            cfg.log_config,
            cfg.get('momentum_config', None),
            custom_hooks_config=cfg.get('custom_hooks', None))

    elif cfg.get('log_config', None) is None and len(cfg.workflow) and cfg.workflow[0][0] != 'simple_train':
        # 提供time, data_time, memory等，并且用于mode里区别IterBasedRunner? 在train模式下提供了有无time的区别
        if cfg.mode == 'nni':
            runner.register_custom_hooks({'type': 'NNIHook', 'priority': 'very_low'})
        if scheduler is not None:
            runner.register_lr_hook(dict(policy=scheduler.__class__.__name__[:-2], step=scheduler.step_size))
        runner.register_checkpoint_hook(
            dict(type='ModelCheckpoint', indicator='loss', save_top_k=cfg.save_top_k, print_freq=10))
        runner.register_optimizer_hook(dict(grad_clip=10))  # ExternOptimizer
        runner.register_timer_hook(dict(type='IterTimerHook'))
        log_config = [dict(type='TextLoggerHook')]
        if cfg.use_tfb:
            log_config.append(dict(type='TensorboardLoggerHook'))
        runner.register_logger_hooks(dict(
            interval=cfg.print_freq,
            hooks=log_config))

    else:
        runner.register_checkpoint_hook(dict(type='ModelCheckpoint', indicator='loss'))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    ############################################################
    # register validate hooks
    ############################################################
    # if cfg.validate:
    #     # Support batch_size > 1 in validation
    #     val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
    #     if val_samples_per_gpu > 1:
    #         # Replace 'ImageToTensor' to 'DefaultFormatBundle'
    #         cfg.data.val.pipeline = replace_ImageToTensor(
    #             cfg.data.val.pipeline)
    #     val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    #     val_dataloader = build_dataloader(
    #         val_dataset,
    #         samples_per_gpu=val_samples_per_gpu,
    #         workers_per_gpu=cfg.data.workers_per_gpu,
    #         dist=distributed,
    #         shuffle=False)
    #     eval_cfg = cfg.get('evaluation', {})
    #     eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
    #     eval_hook = DistEvalHook if distributed else EvalHook
    #     # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
    #     # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
    #     runner.register_hook(
    #         eval_hook(val_dataloader, **eval_cfg), priority='LOW')
    data_loaders = {}


    for idx, flow in enumerate(cfg.workflow):
        mode, epoch = flow
        if 'test' in mode:
            # cfg.dataset = cfg.dataset + '_OrigScale_multiExm1.h5'
            # cfg.dataset = cfg.dataset + '_multiExm1.h5'

            eval_loader, eval_sampler = sess.get_eval_dataloader(cfg.dataset[mode], distributed)

            eval_cfg = cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            from mmcv.runner import EvalHook, DistEvalHook
            eval_hook = DistEvalHook if distributed else EvalHook
            # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
            # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
            if mode != 'simple_val':
                runner.register_hook(
                    eval_hook(eval_loader, **eval_cfg), priority='LOW')

            data_loaders['val'] = eval_loader
            cfg.workflow[idx] = ('val', epoch)
            # if len(cfg.workflow) == 0:
            #     cfg.workflow.append(('val', 1))

        if 'valid' in mode:
            valid_loader, valid_sampler = sess.get_valid_dataloader(cfg.dataset[mode], distributed)
            if cfg.once_epoch:
                valid_loader = iter(list(valid_loader))
            data_loaders['val'] = valid_loader
            cfg.workflow[idx] = ('val', epoch)

        if 'train' in mode:
            train_loader, train_sampler = sess.get_dataloader(cfg.dataset[mode], distributed)
            if cfg.once_epoch:
                train_loader = iter(list(train_loader))
            data_loaders[mode] = train_loader

            if len(cfg.workflow) == 0:
                cfg.workflow.append(('simple_train', 1))
    ############################################################
    # 载入模型
    ############################################################

    resume_from = None
    if cfg.get('resume_from', None) is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    # if cfg.get('resume_from', None):
    runner.resume(cfg.resume_from, cfg.resume_mode, cfg.reset_lr, cfg.lr)
    if cfg.get('load_from', None) and cfg.get('resume_from', None) is not None:
        runner.load_checkpoint(cfg.load_from, cfg.resume_mode)

    ############################################################
    # 载入数据，运行模型
    ############################################################
    runner.run(data_loaders, cfg.workflow)


def main(cfg):
    # init distributed env first, since logger depends on the dist info.
    if cfg.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(cfg.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    logger, out_dir, model_save_dir, tfb_dir = create_logger(cfg, cfg.experimental_desc, 0)
    cfg.out_dir = cfg.work_dir = model_save_dir
    seed = init_random_seed(cfg.seed)
    print_log(f'Set random seed to {seed}', logger=logger)

    set_random_seed(seed)

    # if cfg.checkpoint_config is not None:
    #     # save mmdet version, config file content and class names in
    #     # checkpoints as meta data
    #     cfg.checkpoint_config.meta = dict(
    #         mmdet_version=__version__ + get_git_hash()[:7],
    #         CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience

    trainer(
        cfg,
        logger,
        distributed=distributed,
        meta={})
