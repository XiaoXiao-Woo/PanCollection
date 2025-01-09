# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @reference: 
#
import udl_vis.Basis.option

def build_model(cfg=None):
    from pancollection.models.base_model import PanSharpeningModel as MODELS
    return MODELS.build_model_from_task(cfg)


def getDataSession(cfg):
    from pancollection.common.psdata import PansharpeningSession as DataSession
    return DataSession(cfg)
