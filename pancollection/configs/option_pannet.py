import argparse
from .configs import TaskDispatcher
import os
from .load_ckpt_configs import load_ckpt

class parser_args(TaskDispatcher, name='PanNet'):
    def __init__(self, cfg=None, **kwargs):
        super(parser_args, self).__init__()

        if cfg is None:
            from .configs import panshaprening_cfg
            cfg = panshaprening_cfg()

        script_path = os.path.dirname(os.getcwd())
        root_dir = script_path.split(cfg.task)[0]


        # model_path = f'{root_dir}/results/{cfg.task}/qb_hp/PanNet/Test/.pth.tar'
        model_path = kwargs["model_path"] if 'model_path' in kwargs else f'.pth.tar'
        dataset_name = kwargs["dataset_name"] if 'dataset_name' in kwargs else "wv3"
        use_resume = kwargs["use_resume"] if 'use_resume' in kwargs else True
        model_path = load_ckpt(use_resume=use_resume, model_name="PanNet",
                               dataset_name=dataset_name, model_path=model_path)
        
        parser = argparse.ArgumentParser(description='PyTorch Pansharpening Training')
        # * Logger
        parser.add_argument('--out_dir', metavar='DIR', default=f'{root_dir}/results/{cfg.task}',
                            help='path to save model')
        # * Training
        parser.add_argument('--lr', default=1e-3, type=float)  # 1e-4 2e-4 8
        parser.add_argument('--lr_scheduler', default=True, type=bool)
        parser.add_argument('--samples_per_gpu', default=32, type=int,  # 8
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--print-freq', '-p', default=50, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--seed', default=1, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--epochs', default=450, type=int)
        parser.add_argument('--workers_per_gpu', default=0, type=int)
        parser.add_argument('--resume_from',
                            default=model_path,
                            type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')

        # * Model and Dataset
        parser.add_argument('--arch', '-a', metavar='ARCH', default='PanNet', type=str,
                            choices=['PanNet', 'DiCNN', 'PNN', 'FusionNet'])
        parser.add_argument('--dataset', default={'train': 'wv3', 'test': 'wv3_multiExm1_hp.h5'}, type=str,
                            choices=[None, 'wv2', 'wv3', 'wv4', 'qb', 'gf',
                                     'wv2_hp', ...,
                                     'wv3_OrigScale_multiExm_hp.h5', 'wv3_multiExm1_hp.h5'],
                            help="performing evalution for patch2entire")
        parser.add_argument('--eval', default=False, type=bool,
                            help="performing evalution for patch2entire")

        args = parser.parse_args(args=[])
        args.start_epoch = args.best_epoch = 1
        args.experimental_desc = "Test"


        cfg.merge_args2cfg(args)
        cfg.img_range = 2047.0 if dataset_name != "gf2" else 1023.0
        cfg.reg = True
        cfg.workflow = [('train', 1)]
        cfg.dataloader_name = "PanCollection_dataloader"  # PanCollection_dataloader, oldPan_dataloader, DLPan_dataloader
        cfg.merge_from_dict(kwargs)
        cfg.dataset = kwargs['dataset'] if 'dataset' in kwargs else cfg.dataset
        print(cfg.pretty_text)

        self.merge_from_dict(cfg)
