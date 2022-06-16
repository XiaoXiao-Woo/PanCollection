# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved 
#
# @Time    : 2022/4/25 0:17
# @Author  : Xiao Wu
# @reference:
import sys
sys.path.append('../..')
from UDL.AutoDL import TaskDispatcher
from UDL.AutoDL.trainer import main

if __name__ == '__main__':
    cfg = TaskDispatcher.new(task='pansharpening', mode='entrypoint', arch='BDPN')
    print(TaskDispatcher._task.keys())
    main(cfg)