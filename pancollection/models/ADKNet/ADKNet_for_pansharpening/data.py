import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np


def get_edge(data):
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        data = h5py.File(file_path)

        gt = data["gt"][...]
        gt = np.array(gt, dtype=np.float32) / 2047.
        self.gt = torch.from_numpy(gt)

        lrms = data["ms"][...]
        lrms = np.array(lrms, dtype=np.float32) / 2047.
        self.lrms = torch.from_numpy(lrms)

        pan = data['pan'][...]
        pan = np.array(pan, dtype=np.float32) / 2047.
        self.pan = torch.from_numpy(pan)

    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.lrms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float(),

    def __len__(self):
        return self.gt.shape[0]

