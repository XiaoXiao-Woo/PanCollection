# GPL License
# Copyright (C) 2022 , UESTC
# All Rights Reserved 
#
# @Time    : 2023/3/27 13:40
# @Author  : Xiao Wu
# @reference:
from UDL.Basis.python_sub_class import ModelDispatcher
import numpy as np
import torch
from torch.nn import functional as F
from common.evaluate import analysis_accu
from common.dataset import save_results

class PanSharpeningModel(ModelDispatcher, name='pansharpening'):

    _models = {}

    def __init__(self, model=None, criterion=None):
        super(PanSharpeningModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.reg = False
        self.SAM_list = []
        self.ERGAS_list = []
        if hasattr(self.model, 'reg'):
            self.reg = self.model.reg

    def __init_subclass__(cls, name='', **kwargs):

        # print(name, cls)
        if name != '':
            cls._models[name] = cls
            cls._name = name
        else:
            cls._models[cls.__name__] = cls
            cls._name = cls.__name__
            # warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')

    def l2_regularization(self, loss_dict, weight_decay=1e-5, flag=False):
        regularizations = []
        for k, v in self.model.named_parameters():
            if 'conv' in k and 'weight' in k:
                # print(k)
                penality = weight_decay * ((v.data ** 2).sum() / 2)
                regularizations.append(penality)
                if flag:
                    print("{} : {}".format(k, penality))
        # r = torch.sum(regularizations)
        if isinstance(loss_dict, dict):
            loss_dict['loss'] = loss_dict['loss'] + sum(regularizations)
            loss_dict['log_vars'].update(reg_loss=loss_dict['loss'])
        else:
            loss_dict = loss_dict + sum(regularizations)

        return loss_dict

    def train_step(self, *args, **kwargs):

        loss_dict = self.model.train_step(args[0], **kwargs)

        if self.reg:
            return self.l2_regularization(loss_dict)

        return loss_dict

    def val_step(self, *args, **kwargs):
        sr, gt = self.model.val_step(*args, **kwargs)
        if kwargs['eval']:
            result_our = torch.squeeze(sr).permute(1, 2, 0)
            metrics = analysis_accu(gt.cuda().squeeze(0), result_our, 4)
            result_our = result_our * kwargs['img_range']
            if kwargs['idx'] not in [220, 231, 236, 469, 766, 914]:
                if kwargs['save_fmt'] is not None:
                    save_results(kwargs['idx'], kwargs['save_dir'], kwargs['filename'], kwargs['save_fmt'], result_our)
                self.SAM_list.append(metrics['SAM'].item())
                self.ERGAS_list.append(metrics['ERGAS'].item())
            if kwargs['idx'] == 1257:
                print(np.mean(self.SAM_list), np.mean(self.ERGAS_list))
        else:
            sr, gt = self.model.val_step(*args, **kwargs)
            metrics = {'loss': F.l1_loss(sr, gt).item()}

        return {'log_vars': metrics}

