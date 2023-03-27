import argparse
# from UDL.Basis.option import panshaprening_cfg, Config, os
from UDL.AutoDL import TaskDispatcher
import os

class parser_args(TaskDispatcher, name='FusionNet'):
    def __init__(self, cfg=None):
        super(parser_args, self).__init__()

        if cfg is None:
            from UDL.Basis.option import panshaprening_cfg
            cfg = panshaprening_cfg()

        script_path = os.path.dirname(os.path.dirname(__file__))
        root_dir = script_path.split(cfg.task)[0]

        model_path = f'.pth.tar'


        parser = argparse.ArgumentParser(description='PyTorch Pansharpening Training')
        # * Logger
        parser.add_argument('--out_dir', metavar='DIR', default=f'{root_dir}/results/{cfg.task}',
                            help='path to save model')
        parser.add_argument('--mode', default=argparse.SUPPRESS, help='protective declare, please ignore it')

        parser.add_argument('--lr', default=3e-4, type=float)  # 1e-4 2e-4 8
        # parser.add_argument('--lr_scheduler', default=True, type=bool)
        parser.add_argument('--samples_per_gpu', default=32, type=int,  # 8
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--print-freq', '-p', default=50, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--seed', default=1, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--epochs', default=400, type=int)
        parser.add_argument('--workers_per_gpu', default=0, type=int)
        parser.add_argument('--resume_from',
                            default=model_path,
                            type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        ##
        parser.add_argument('--arch', '-a', metavar='ARCH', default='FusionNet', type=str,
                            choices=['PanNet', 'DiCNN', 'PNN', 'FusionNet'])
        parser.add_argument('--dataset', default={'train': 'wv3', 'test': 'test_wv3_multiExm1.h5'}, type=str, # 'valid': 'wv3' , 'eval': 'wv3_multiExm.h5'
                            choices=[None, 'wv2', 'wv3', 'wv4', 'qb', 'gf2',
                                     'wv3_OrigScale_multiExm1.h5', 'test_wv3_multiExm1.h5'],
                            help="performing evalution for patch2entire")
        parser.add_argument('--eval', default=True, type=bool,
                            help="performing evaluation out of training process, which can avoid dead loop !!")


        args = parser.parse_args()
        args.start_epoch = args.best_epoch = 1
        args.experimental_desc = 'Test'
        cfg.merge_args2cfg(args)
        cfg.save_fmt = "mat"
        # cfg.workflow = [('train', 10), ('valid', 1)]
        cfg.workflow = [('valid', 1)]
        # cfg.config = f"{script_path}/configs/hook_configs.py"
        cfg.use_tfb = False
        cfg.img_range = 2047.0#1023.0
        cfg.dataloader_name = "PanCollection_dataloader"  # PanCollection_dataloader, oldPan_dataloader, DLPan_dataloader


        cfg.merge_args2cfg(args)

        self.merge_from_dict(cfg)
