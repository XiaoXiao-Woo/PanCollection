# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @reference: 
#
import udl_vis.Basis.option

def build_model(arch, task, cfg=None):

    if task == "pansharpening":
        from pancollection.models.base_model import PanSharpeningModel as MODELS

        return MODELS.build_model(cfg)
    else:
        raise NotImplementedError(f"It's not supported in {task}")


def getDataSession(cfg):

    task = cfg.task

    if task in ["pansharpening"]:
        from pancollection.common.psdata import PansharpeningSession as DataSession
    else:
        raise NotImplementedError

    return DataSession(cfg)
