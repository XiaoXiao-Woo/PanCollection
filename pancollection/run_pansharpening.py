# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu
# @reference:

def run_demo(**kwargs):
    from pancollection import TaskDispatcher, trainer, build_model, getDataSession
    cfg = TaskDispatcher.new(task='pansharpening', mode='entrypoint', arch=kwargs.pop('arch'),
                             **kwargs
                             )
    print(TaskDispatcher._task.keys())
    trainer.main(cfg, build_model, getDataSession)
    # or
    # import pancollection as pan
    # cfg = pan.TaskDispatcher.new(task='pansharpening', mode='entrypoint', arch='FusionNet')
    # print(pan.TaskDispatcher._task)
    # pan.trainer.main(cfg, pan.build_model, pan.getDataSession)

if __name__ == '__main__':
    arch = 'PNN'
    dataset_name = 'gf2'
    cfg = dict(arch=arch, dataset_name=dataset_name, use_resume=False,
                      dataset={'train': 'gf2', 'test': 'test_gf2_multiExm1.h5'})
    run_demo(**cfg)