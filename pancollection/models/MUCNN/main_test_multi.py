# import torch.nn.modules as nn
import numpy as np
import torch
from model.Model import Unet
import h5py
import scipy.io as io
import cv2


def get_edge(data):  # for training
    rs = np.zeros_like(data)
    _N = data.shape[0]

    for i in range(_N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs


def load_set(file__path):
    print(file__path)
    data = h5py.File(file__path)  # NxHxWxC

    # tensor type: NxCxHxW:
    lms = np.array(data.get('lms'))
    lms = torch.from_numpy(lms / 2047).permute(0, 3, 1, 2)

    ms_hp = torch.from_numpy((np.array(data.get('ms')) / 2047)).permute(0, 3, 1, 2)  # NxCxHxW:

    pan = np.array(data.get('pan'))
    # pan_tmp = pan[:, :, :, np.newaxis]  # NxHxWxC (C=1)
    pan_hp = torch.from_numpy((pan / 2047)[:, :, :, np.newaxis]).permute(0, 3, 1, 2)  # Nx1xHxW:

    return lms, ms_hp, pan_hp

def load_set_mat(file__path):
    data = io.loadmat(file__path)  # HxWxC

    # tensor type:
    lms = torch.from_numpy(data['lms'] / 2047).permute(0, 3, 1, 2)  # CxHxW = 8x256x256
    ms_hp = torch.from_numpy((data['ms'] / 2047)).permute(0, 3, 1, 2)  # CxHxW= 8x64x64
    pan_hp = torch.from_numpy((data['pan'] / 2047)[:, :, :, np.newaxis]).permute(0, 3, 1, 2)    # HxW = 256x256
    return lms, ms_hp, pan_hp

# ==============  Main test  ================== #


ckp = "Weight/model_0121/600.pth"


def test(file__path, use_cuda=1):
    lms, ms_hp, pan_hp = load_set(file__path)

    if use_cuda:
        model = Unet(1, 8).cuda().eval()
    else:
        model = Unet(1, 8).eval()
    if use_cuda:
        weight = torch.load(ckp)
        model.load_state_dict(weight)
    else:
        weight = torch.load(ckp, map_location='cpu')
        model.load_state_dict(weight)

    with torch.no_grad():
        for i in range(ms_hp.shape[0]):
            x1, x2, x3 = lms[i, :, :, :], ms_hp[i, :, :, :], pan_hp[i, :, :, :]   # read data: CxHxW (numpy type)

            if use_cuda:
                x1 = x1.cuda().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (un squeeze(dim=0))
                x2 = x2.cuda().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW
                x3 = x3.cuda().unsqueeze(dim=0).float()  # convert to tensor type: 1x1xHxW
            else:
                x1 = x1.unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (un squeeze(dim=0))
                x2 = x2.unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW
                x3 = x3.unsqueeze(dim=0).float()  # convert to tensor type: 1x1xHxW

            aux1, aux2, hp_sr = model(x2, x3)  # tensor type: 1xCxHxW
            sr = hp_sr        # tensor type: 1xCxHxW
            # sr = x1 + hp_sr        # tensor type: 1xCxHxW

            # convert to numpy type with permute and squeeze: HxWxC (go to cpu for easy saving)
            sr = torch.squeeze(sr).permute(1, 2, 0).cpu().detach().numpy()

            filename = "your file path"
            io.savemat(filename, {'sr': sr})
            print('{}/{}'.format(i, ms_hp.shape[0]))


if __name__ == '__main__':
    """@key: Absolute path"""
    file_path = "your file path"
    test(file_path, 0)
