import argparse
from configs.configs import TaskDispatcher
import os

class parser_args(TaskDispatcher, name='DRPNN'):
    def __init__(self, cfg=None):
        super(parser_args, self).__init__()

        if cfg is None:
            from UDL.Basis.option import panshaprening_cfg
            cfg = panshaprening_cfg()

        script_path = os.path.dirname(os.path.dirname(__file__))
        root_dir = script_path.split(cfg.task)[0]

        model_path = f'{root_dir}/results/{cfg.task}/wv3/DRPNN/Test/.pth.tar'

        parser = argparse.ArgumentParser(description='PyTorch Pansharpening Training')
        # * Logger
        parser.add_argument('--out_dir', metavar='DIR', default=f'{root_dir}/results/{cfg.task}',
                            help='path to save model')
        # * Training
        parser.add_argument('--lr', default=2e-4, type=float)  # 1e-4 2e-4 8
        parser.add_argument('--lr_scheduler', default=True, type=bool)
        parser.add_argument('--samples_per_gpu', default=64, type=int,  # 8
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--print-freq', '-p', default=50, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--epochs', default=3000, type=int)
        parser.add_argument('--workers_per_gpu', default=0, type=int)
        parser.add_argument('--resume_from',
                            default=model_path,
                            type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        # * Model and Dataset
        parser.add_argument('--arch', '-a', metavar='ARCH', default='DRPNN', type=str,
                            choices=['PanNet', 'DiCNN', 'PNN', 'FusionNet'])
        # _multiExm1.h5
        parser.add_argument('--dataset', default={'train': 'wv3', 'test': 'wv3_multiExm1.h5'}, type=str,
                            choices=[None, 'wv2', 'wv3', 'wv4', 'qb', 'gf',
                                     'wv3_OrigScale_multiExm1.h5', 'wv3_multiExm1.h5'],
                            help="performing evalution for patch2entire")
        parser.add_argument('--eval', default=False, type=bool,
                            help="performing evalution for patch2entire")

        args = parser.parse_args()
        args.start_epoch = args.best_epoch = 1
        args.experimental_desc = "Test"
        # cfg.save_fmt = 'png'
        cfg.img_range = 2047.0
        cfg.dataloader_name = "PanCollection_dataloader"  # PanCollection_dataloader, oldPan_dataloader, DLPan_dataloader

        cfg.merge_args2cfg(args)
        print(cfg.pretty_text)
        # cfg.workflow = [('train', 50), ('val', 1)]
        cfg.workflow = [('valid', 1)]
        self.merge_from_dict(cfg)

