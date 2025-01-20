import torch.utils.data as data
import scipy.io as sio
import numpy as np
import torch
import h5py
import cv2

def get_edge(data):  # for training
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        data = h5py.File(file_path)

        GT = data["gt"][...]
        GT = GT.transpose(0, 3, 1, 2)
        GT = np.array(GT, dtype=np.float32)/(2**16-1)
        self.GT = GT

        LRHS = data["ms"][...]
        LRHS = LRHS.transpose(0, 3, 1, 2)
        LRHS = np.array(LRHS, dtype=np.float32) / (2 ** 16 - 1)
        self.LRHS = LRHS

        RGB = data["rgb1"][...]
        RGB = RGB.transpose(0, 3, 1, 2)
        RGB = np.array(RGB, dtype=np.float32) / (2**8-1)
        self.RGB = RGB


    def __getitem__(self, index):
        return torch.from_numpy(self.GT[index, :, :, :]).float(), \
               torch.from_numpy(self.LRHS[index, :, :, :]).float(), \
               torch.from_numpy(self.RGB[index, :, :, :]).float(),


    def __len__(self):
        return self.GT.shape[0]