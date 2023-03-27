import platform
import warnings
from UDL.Basis.option import common_cfg
from UDL.Basis.python_sub_class import TaskDispatcher


class panshaprening_cfg(TaskDispatcher, name='pansharpening'):


    def __init__(self, cfg=None, arch=None):
        super(panshaprening_cfg, self).__init__()

        import configs
        import models

        if cfg is None:
            cfg = common_cfg()

        cfg.scale = [1]
        if platform.system() == 'Linux':
            cfg.data_dir = '/Data/Datasets/pansharpening_2'
        if platform.system() == "Windows":
            cfg.data_dir = 'D:/Datasets/pansharpening'

        cfg.best_prec1 = 10000
        cfg.best_prec5 = 10000
        cfg.metrics = 'loss'
        cfg.task = "pansharpening"
        cfg.save_fmt = "mat" # fmt is mat or not mat
        cfg.taskhead = "pansharpening"

        # * Importantly
        warning = f"Note: FusionNet, DiCNN, PNN don't have high-pass filter"
        warnings.warn(warning)
        if arch is not None:
            cfg = self.new(cfg=cfg, arch=cfg.arch)
        self.merge_from_dict(cfg)