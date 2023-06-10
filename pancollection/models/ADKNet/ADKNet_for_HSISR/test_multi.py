import h5py
import torch
import numpy as np
import scipy.io as sio
from model import ADKNet


def load_set(file_path):

    data = h5py.File(file_path)
    GT = data["gt"][...]
    GT = GT.transpose(0, 3, 1, 2)
    GT = np.array(GT, dtype=np.float32) / (2 ** 16 - 1)
    GT = torch.from_numpy(GT)

    LRHS = data["ms"][...]
    LRHS = LRHS.transpose(0, 3, 1, 2)
    LRHS = np.array(LRHS, dtype=np.float32) / (2 ** 16 - 1)
    LRHS = torch.from_numpy(LRHS)

    RGB = data["rgb1"][...]
    RGB = RGB.transpose(0, 3, 1, 2)
    RGB = np.array(RGB, dtype=np.float32) / (2 ** 8 - 1)
    RGB = torch.from_numpy(RGB)

    return GT, LRHS, RGB


ckpt = "Weights/ADKNet4HSISR.pth"


def test(file_path):
    _, LRHS, RGB = load_set(file_path)
    model = ADKNet().cuda().eval()
    weight = torch.load(ckpt)
    model.load_state_dict(weight)
    output = np.zeros([LRHS.shape[0], RGB.shape[2], RGB.shape[3], 31])

    with torch.no_grad():
        for i in range(LRHS.shape[0]):
            x1, x2 = LRHS[i, :, :, :], RGB[i, :, :, :]
            x1 = x1.cuda().unsqueeze(dim=0).float()
            x2 = x2.cuda().unsqueeze(dim=0).float()

            with torch.no_grad():
                sr = model(x1, x2)
                sr = torch.clamp(sr, 0, 1)
                output[i, :, :, :] = sr.permute([0, 2, 3, 1]).cpu().detach().numpy()

    sio.savemat('results/cave_ADKNet.mat', {'output': output})
    #sio.savemat('results/harvard_ADKNet.mat', {'output': output})


if __name__ == '__main__':
    """@key: Absolute path"""
    file_path = "test_data/test_cave.mat"
    #file_path = "test_data/test_harvard.mat"
    test(file_path)

