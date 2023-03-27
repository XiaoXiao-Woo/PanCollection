# GPL License
# Copyright (C) UESTC
# All Rights Reserved 
#
# @Time    : 2022/4/25 0:17
# @Author  : Xiao Wu
# @reference:
import sys
sys.path.append('../..')
from configs.configs import TaskDispatcher
from UDL.AutoDL.trainer import main
from common.builder import build_model, getDataSession

if __name__ == '__main__':
    cfg = TaskDispatcher.new(task='pansharpening', mode='entrypoint', arch='BDPN')
    print(TaskDispatcher._task.keys())
    main(cfg, build_model, getDataSession)