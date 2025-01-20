# GPL License
# Copyright (C) UESTC
# All Rights Reserved 
#
# @Time    : 2023/6/4 15:08
# @Author  : Xiao Wu
# @reference: 
#
import argparse
import sys
from udl_vis.Basis.option import TaskDispatcher
import os

class parser_args(TaskDispatcher, name='DPM_ps'):
    def __init__(self, cfg=None):
        super(parser_args, self).__init__()

        if cfg is None:
            from pancollection.configs.configs import panshaprening_cfg
            cfg = panshaprening_cfg()

        script_path = os.path.dirname(os.path.dirname(__file__))
        root_dir = script_path.split(cfg.task)[0].replace('\\', '/')

        ckpt_model_path = ""
        
        test_model_path = "/Data3/YuZhong/SSdiff_main/SSdiff_main-main/results/ema_0.9999_021000.pt"


        parser = argparse.ArgumentParser(description='PyTorch Training')
        # * Logger
        parser.add_argument('--out_dir', metavar='DIR', default=f'{root_dir}/results/{cfg.task}',
                            help='path to save model')
        # * Training
        parser.add_argument('--lr', default=1e-3, type=float)  # 1e-4 2e-4
        parser.add_argument('--lr_scheduler', default=True, type=bool)
        parser.add_argument('--crop_batch_size', default=24, type=int)
        parser.add_argument('--samples_per_gpu', default=20, type=int,              # batch_size
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--print-freq', '-p', default=500, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--seed', default=3407, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--epochs', default=12000, type=int) # 12000
        parser.add_argument('--workers_per_gpu', default=0, type=int)
        parser.add_argument('--device', default='cpu', type=str)
        parser.add_argument('--resume_from',
                            default=ckpt_model_path,
                            type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        # * Model and Dataset
        parser.add_argument('--arch', '-a', metavar='ARCH', default='iDPM', type=str,
                            choices=['PanNet', 'DiCNN', 'PNN', 'FusionNet'])
        parser.add_argument('--dataset', default={'train': 'wv3', 'val': 'wv3', 'test': 'test_wv3_multiExm1.h5'}, type=str,
                            choices=[None, 'wv2', 'wv3', 'wv4', 'qb', 'gf2',
                                     'wv3_OrigScale_multiExm1.h5', 'wv3_multiExm1.h5'],
                            help="performing evalution for patch2entire")
        parser.add_argument('--eval', default=False, type=bool,
                            help="performing evalution for patch2entire")

        parser.add_argument('--dim', default=32, type=int)
        parser.add_argument('--dim_head', default=16, type=int)
        parser.add_argument('--se_ratio_mlp', default=0.5, type=float)
        parser.add_argument('--se_ratio_rb', default=0.5, type=float)
        parser.add_argument('--ms_dim', default=8, type=int) # qb/gf2:4,  wv3:8
        parser.add_argument('--pan_dim', default=1, type=int)
        parser.add_argument('--model_channels', default=128, type=int)
        
        
        # * DPM
        parser.add_argument('--schedule_sampler', default="uniform"),
        parser.add_argument('--lr_anneal_steps', default=0),
        parser.add_argument('--microbatch', default=-1),  # -1 disables microbatches
        parser.add_argument('--ema_rate', default="0.9999"),  # comma-separated list of EMA values
        parser.add_argument('--use_fp16', default=False),
        parser.add_argument('--fp16_scale_growth', default=1e-3),

        parser.add_argument('--image_size', default=64)
        parser.add_argument('--num_channels', default=128)  # 128

        parser.add_argument('--num_res_blocks', default=2)
        parser.add_argument('--num_heads', default=4)
        parser.add_argument('--num_heads_upsample', default=-1)
        parser.add_argument('--attention_resolutions', default="16,8")
        parser.add_argument('--dropout', default=0.0)
        parser.add_argument('--learn_sigma', default=False)
        parser.add_argument('--sigma_small', default=False)
        parser.add_argument('--class_cond', default=False)
        parser.add_argument('--diffusion_steps', default=1000)
        parser.add_argument('--noise_schedule', default="cosine")
        parser.add_argument('--timestep_respacing', default="ddim1000")
        parser.add_argument('--use_kl', default=False)
        parser.add_argument('--predict_xstart', default=True)
        parser.add_argument('--rescale_timesteps', default=True)
        parser.add_argument('--rescale_learned_sigmas', default=True)
        parser.add_argument('--use_scale_shift_norm', default=False)
        parser.add_argument('--use_checkpoint', default=False)

        # *
        parser.add_argument('--log_interval', default=100)
        parser.add_argument('--save_interval', default=1000)
        parser.add_argument('--resume_checkpoint', default="")
        parser.add_argument('--weight_decay', default=0)

        # * sample
        parser.add_argument('--clip_denoised', default=True)
        parser.add_argument('--num_samples', default=2047)
        parser.add_argument('--test_samples_per_gpu', default=16)
        parser.add_argument('--use_ddim', default=True)
        parser.add_argument('--model_path', default=test_model_path) # model_path

        args = parser.parse_args()
        args.start_epoch = args.best_epoch = 1
        args.experimental_desc = 'Test'
        cfg.merge_args2cfg(args)
        cfg.workflow = [('train', 1)]
        cfg.img_range = 2047.0
        cfg.dataloader_name = "PanCollection_dataloader"

        self.merge_from_dict(cfg)
