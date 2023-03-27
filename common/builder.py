# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved 
#
# @Time    : 2022/9/16 22:10
# @Author  : Xiao Wu
# @reference: 
#
import UDL.Basis.option

def build_model(arch, task, cfg=None):

    if task == "pansharpening":
        from models.base_model import PanSharpeningModel as MODELS

        return MODELS.build_model(cfg)
    else:
        raise NotImplementedError(f"It's not supported in {task}")


def getDataSession(cfg):

    task = cfg.task

    if task in ["pansharpening"]:
        from common.psdata import PansharpeningSession as DataSession
    else:
        raise NotImplementedError

    return DataSession(cfg)
