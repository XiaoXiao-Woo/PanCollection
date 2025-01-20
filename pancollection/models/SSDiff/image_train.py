"""
Train a diffusion model on images.

Matlab test code for WV3:

cd Matlab-Test-Package
analysis_ref_batched_images('*.mat', 4, 0, 2047)

"""
import os
import torch.cuda
import torch as th
import numpy as np

rootPath = os.path.abspath(os.path.dirname(__file__))

from improved_diffusion import logger
from improved_diffusion.resample import create_named_schedule_sampler
from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

from utils.train_util import TrainLoop
from pancollection.common.psdata import PansharpeningSession as DataSession
from configs.option_DPM_pansharpening import parser_args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def main(
    device='cuda:0',
    Resume = False
    ):
    
    args = parser_args()
    set_seed(1)
    if device is not None:
        args.device = device
    torch.cuda.set_device(args.device)
    logger.configure(dir='/'.join([rootPath, 'logs/train_logs/']))

    logger.log("creating model and diffusion...")
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    
    
    if Resume:
        path_checkpoint = args.resume_from
        logger.log("path_checkpoint: ", path_checkpoint)
        model.load_state_dict(torch.load(path_checkpoint, map_location=lambda storage, loc: storage.cuda()), strict=False)

    session = DataSession(args)
    
    """  pansharpening  """
    data, _ , _= session.get_dataloader(args.dataset['train'], False, None)    
 
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        device=args.device,
        batch_size=args.samples_per_gpu,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


if __name__ == "__main__":

    main()
