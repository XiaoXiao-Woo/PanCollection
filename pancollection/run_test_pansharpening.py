# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu
# @reference:

def run_test(**kwargs):
    arch = kwargs.pop('arch')
    from pancollection import TaskDispatcher, trainer, build_model, getDataSession
    cfg = TaskDispatcher.new(task='pansharpening', mode='entrypoint', arch=arch,
                             eval=kwargs.pop('eval'), **kwargs) # 'test_wv3_OrigScale_multiExm1.h5'
    print(TaskDispatcher._task.keys())
    trainer.main(cfg, build_model, getDataSession)

def run_test1(arch, dataset_name):
    from pancollection import TaskDispatcher, trainer, build_model, getDataSession
    cfg = TaskDispatcher.new(task='pansharpening', mode='entrypoint', arch=arch, dataset_name=dataset_name)
    cfg.eval = True  # test model
    cfg.dataset = {'test': f'test_{dataset_name}_multiExm1_hp.h5'} \
        if arch == "PanNet" else {'test': f'test_{dataset_name}_multiExm1.h5'} # 'test_wv3_OrigScale_multiExm1.h5'
    print(TaskDispatcher._task.keys())
    trainer.main(cfg, build_model, getDataSession)
    # or
    # import pancollection as pan
    # cfg = pan.TaskDispatcher.new(task='pansharpening', mode='entrypoint', arch='FusionNet')
    # ...
    # print(pan.TaskDispatcher._task)
    # pan.trainer.main(cfg, pan.build_model, pan.getDataSession)


def run_test2(arch, dataset_name):
    from pancollection import TaskDispatcher, trainer, build_model, getDataSession
    cfg = TaskDispatcher.new(task='pansharpening', mode='entrypoint', arch=arch,
                             dataset_name=dataset_name, eval=True, model_path="D:/Python/gitSync/AEM/ckpt/PanCollection/wv3/FusionNet/FusionNet.pth.tar",
                             dataset={'test': f'test_{dataset_name}_multiExm1_hp.h5'}
                             if arch == "PanNet" else {'test': f'test_{dataset_name}_multiExm1.h5'}) # 'test_wv3_OrigScale_multiExm1.h5'
    print(TaskDispatcher._task.keys())
    trainer.main(cfg, build_model, getDataSession)


def run_test3(arch, dataset_name):
    import pancollection as pan
    cfg = pan.configs.option_dicnn.parser_args(dataset_name=dataset_name,
                                                   eval=True,
                                                   dataset={'train': 'gf2', 'test': 'test_gf2_multiExm1.h5'})
    print(pan.TaskDispatcher._task)
    pan.trainer.main(cfg, pan.build_model, pan.getDataSession)


if __name__ == '__main__':
    dataset_name = "wv3"
    # run_test2("PNN", dataset_name)
    # run_test2("DiCNN", dataset_name)
    run_test2("FusionNet", dataset_name)
    # run_test2("PanNet", dataset_name)
    # run_test2("LAGNet", dataset_name)

    # run_test("FusionNet", dataset_name)
    # run_test1("FusionNet", dataset_name)
    # run_test2("DiCNN", dataset_name)
