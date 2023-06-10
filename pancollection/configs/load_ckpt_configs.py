import glob
import os


def load_ckpt(use_resume, model_name, model_path, root_path = "D:/Python/gitSync/ckpts/PanCollection",
              dataset_name="wv3", ):

    if not os.path.isfile(model_path) and use_resume:
        tmp = glob.glob('/'.join([root_path, dataset_name, model_name, '*.pth.tar']))[0]
        if os.path.isfile(tmp):
            model_path = tmp
    # PNN_PATH = glob.glob('/'.join(root_path, dataset_name, "PNN", '*.pth.tar'))
    # DiCNN_PATH = glob.glob('/'.join(root_path, dataset_name, "DiCNN", '*.pth.tar'))
    # PanNet_PATH = glob.glob('/'.join(root_path, dataset_name, "PanNet", '*.pth.tar'))
    # FusionNet_PATH = glob.glob('/'.join(root_path, dataset_name, "FusionNet", '*.pth.tar'))
    # LAGNet_PATH = glob.glob('/'.join(root_path, dataset_name, "LAGNet", '*.pth.tar'))

    return model_path