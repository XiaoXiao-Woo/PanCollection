# checkpoint saving
# checkpoint_config = dict(interval=1)

use_log_and_save = False
save_top_k = 1
save_interval = 20
log_interval = 20
grad_clip = 10

checkpoint_config = dict(type='ModelCheckpoint', indicator='loss', save_top_k=save_top_k,
                 use_log_and_save=use_log_and_save, save_interval=save_interval)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

# dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = ""
resume_from = ""
workflow = [('train', 1)]

# optimizer
optimizer = dict(type='Adam', lr=3e-4)
optimizer_config = dict(grad_clip=None)
lr_config = None
# learning policy
runner = dict(type='EpochBasedRunner', max_epochs=450)
