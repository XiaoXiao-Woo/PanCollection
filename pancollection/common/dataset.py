import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np
import os
# import datetime
# import imageio
import torch.nn.functional as F
from scipy import io as sio
from torch.utils.data import Dataset
from udl_vis.Basis.postprocess import showimage8
import matplotlib.pyplot as plt
# from UDL.Basis.zoom_image_region import show_region_images
# from logging import info as log_string

class Dataset_Pro(data.Dataset):
    def __init__(self, file_path, img_scale):
        super(Dataset_Pro, self).__init__()

        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        print(f"loading Dataset_Pro: {file_path} with {img_scale}, keys: {data.keys()}")
        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / img_scale
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / img_scale

        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / img_scale
        self.lms = torch.from_numpy(lms1)


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / img_scale # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
        print(pan1.shape, lms1.shape, gt1.shape, ms1.shape)
    #####必要函数
    def __getitem__(self, index):
        return {'gt':self.gt[index, :, :, :].float(),
               'lms':self.lms[index, :, :, :].float(),
               'ms':self.ms[index, :, :, :].float(),
               'pan':self.pan[index, :, :, :].float()}

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]



# dmd
def load_gt_compared(file_path_gt, file_path_compared):
    data1 = sio.loadmat(file_path_gt)  # HxWxC
    data2 = sio.loadmat(file_path_compared)
    try:
        gt = torch.from_numpy(data1['gt'] / 2047.0)
    except KeyError:
        print(data1.keys())
    compared_data = torch.from_numpy(data2['output_dmdnet_newdata6'] * 2047.0)
    return gt, compared_data


def get_edge(data):  # get high-frequency
    rs = np.zeros_like(data)
    if rs.ndim == 4:
        for b in range(data.shape[0]):
            for i in range(data.shape[1]):
                rs[b, i, :, :] = data[b, i, :, :] - cv2.boxFilter(data[b, i, :, :], -1, (5, 5))
    elif len(rs.shape) == 3:
        for i in range(data.shape[2]):
            rs[:, :, i] = data[:, :, i] - cv2.boxFilter(data[:, :, i], -1, (5, 5))
    else:
        rs = data - cv2.boxFilter(data, -1, (5, 5))

    return rs


def load_dataset_singlemat_hp(file_path, scale):
    data = sio.loadmat(file_path)  # HxWxC
    print(data.keys())
    # tensor type:
    lms = torch.from_numpy(data['lms'] / scale).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms_hp = torch.from_numpy(get_edge(data['ms'] / scale)).permute(2, 0, 1).unsqueeze(dim=0)  # CxHxW= 8x64x64
    mms_hp = F.interpolate(ms_hp, size=(ms_hp.size(2) * 2, ms_hp.size(3) * 2),
                        mode="bilinear", align_corners=True)
    pan_hp = torch.from_numpy(get_edge(data['pan'] / scale))   # HxW = 256x256
    gt = torch.from_numpy(data['gt'] / scale)

    return lms.squeeze().float(), mms_hp.squeeze().float(), ms_hp.squeeze().float(), pan_hp.float(), gt.float()


def load_dataset_singlemat(file_path, scale):
    data = sio.loadmat(file_path)  # HxWxC
    print(data.keys())
    # tensor type:
    lms = torch.from_numpy(data['lms'] / scale).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms = torch.from_numpy(data['ms'] / scale).permute(2, 0, 1).unsqueeze(dim=0)  # CxHxW= 8x64x64
    mms = F.interpolate(ms, size=(ms.size(2) * 2, ms.size(3) * 2),
                        mode="bilinear", align_corners=True)
    pan = torch.from_numpy(data['pan'] / scale)  # HxW = 256x256
    if data.get('gt', None) is None:
        gt = torch.from_numpy(data['lms'] / scale)
    else:
        gt = torch.from_numpy(data['gt'] / scale)

    return lms.squeeze().float(), mms.squeeze().float(), ms.squeeze().float(), pan.float(), gt.float()


def load_dataset_H5_hp(file_path, scale, use_cuda=True):
    data = h5py.File(file_path)  # NxHxWxC
    shape_list = []
    # for k in data.keys():
    #     shape_list.append((k, data[k].shape))
    # print(shape_list)

    # tensor type: NxCxHxW:

    lms = torch.from_numpy(data['lms'][...] / scale).float()#.permute(0, 3, 1, 2)
    ms_hp = torch.from_numpy(get_edge(data['ms'][...] / scale)).float()#.permute(0, 3, 1, 2)  # NxCxHxW:
    mms_hp = torch.nn.functional.interpolate(ms_hp, size=(ms_hp.size(2) * 2, ms_hp.size(3) * 2),
                                          mode="bilinear", align_corners=True)
    pan = np.squeeze(data['pan'][...])
    pan = pan[:, np.newaxis, :, :]  # NxCxHxW (C=1)
    pan_hp = torch.from_numpy(get_edge(pan / scale)).float()#.permute(0, 3, 1, 2)  # Nx1xHxW:
    if data.get('gt', None) is None:
        gt = torch.from_numpy(data['lms'][...] / scale).float()
    else:
        gt = torch.from_numpy(data['gt'][...] / scale).float()

    return {'lms': lms,
            'mms:': mms_hp,
            'ms': ms_hp,
            'pan': pan_hp,
            'gt': gt.permute([0, 2, 3, 1])
            }

def load_dataset_H5(file_path, scale, suffix, use_cuda=True):
    if suffix == '.h5':
        data = h5py.File(file_path)  # CxHxW
        print(data.keys())
    else:
        data = sio.loadmat(file_path[0])

    # tensor type:
    if use_cuda:
        lms = torch.from_numpy(data['lms'][...] / scale).cuda().float()  # CxHxW = 8x64x64

        ms = torch.from_numpy(data['ms'][...] / scale).cuda().float()  # CxHxW= 8x64x64
        mms = torch.nn.functional.interpolate(ms, size=(ms.size(2) * 2, ms.size(3) * 2),
                                              mode="bilinear", align_corners=True)
        pan = torch.from_numpy(data['pan'][...] / scale).cuda().float()  # HxW = 256x256

        gt = torch.from_numpy(data['gt'][...] / scale).cuda().float()

    else:
        lms = torch.from_numpy(data['lms'][...] / scale).float()  # CxHxW = 8x64x64

        ms = torch.from_numpy(data['ms'][...] / scale).float()  # CxHxW= 8x64x64
        mms = torch.nn.functional.interpolate(ms, size=(ms.size(2) * 2, ms.size(3) * 2),
                                              mode="bilinear", align_corners=True)
        pan = torch.from_numpy(data['pan'][...] / scale).float()  # HxW = 256x256

        if data.get('gt', None) is None:
            gt = torch.from_numpy(data['lms'][...] / scale).float()
        else:
            if np.max(data['gt'][...]) > 1:
                gt = torch.from_numpy(data['gt'][...] / scale).float()
            else:
                gt = torch.from_numpy(data['gt'][...]).float()

    return {'lms': lms,
            'mms:': mms,
            'ms': ms,
            'pan': pan,
            'gt': gt.permute([0, 2, 3, 1])
            }


class MultiExmTest_h5(Dataset):

    def __init__(self, file_path, dataset_name, img_scale, suffix='.h5'):
        super(MultiExmTest_h5, self).__init__()

        # self.scale = 2047.0
        # if 'gf' in dataset_name:
        #     self.scale = 1023.0
        self.img_scale = img_scale
        print(f"loading MultiExmTest_h5: {file_path} with {img_scale}")
        # 一次性载入到内存
        if 'hp' not in dataset_name:
            data = load_dataset_H5(file_path, img_scale, suffix, False)

        elif 'hp' in dataset_name:
            file_path = file_path.replace('_hp', '')
            data = load_dataset_H5_hp(file_path, img_scale, False)

        else:
            print(f"{dataset_name} is not supported in evaluation")
            raise NotImplementedError

        if suffix == '.mat':
            self.lms = data['lms'].permute(0, 3, 1, 2)  # CxHxW = 8x256x256
            self.ms = data['ms'].permute(0, 3, 1, 2)  # CxHxW= 8x64x64
            self.mms = torch.nn.functional.interpolate(self.ms, size=(self.ms.size(2) * 2, self.ms.size(3) * 2),
                                                       mode="bilinear", align_corners=True)
            self.pan = data['pan'].unsqueeze(1)
            self.gt = data['gt'].permute(0, 3, 1, 2)
        else:
            self.lms = data['lms']
            self.ms = data['ms']
            self.mms = torch.nn.functional.interpolate(self.ms, size=(self.ms.size(2) * 2, self.ms.size(3) * 2),
                                                       mode="bilinear", align_corners=True)
            self.pan = data['pan']
            self.gt = data['gt']

        print(f"lms: {self.lms.shape}, ms: {self.ms.shape}, pan: {self.pan.shape}, gt: {self.gt.shape}")

    def __getitem__(self, item):
        return {'lms': self.lms[item, ...],
                'mms': self.mms[item, ...],
                'ms': self.ms[item, ...],
                'pan': self.pan[item, ...],
                'gt': self.gt[item, ...]
                }

    def __len__(self):
        return self.gt.shape[0]


class SingleDataset(Dataset):



    def __init__(self, file_lists, dataset_name, img_scale, dataset=None):
        if dataset is None:
            dataset = ["new_data10", "new_data11", "new_data12_512",
                       "new_data3_wv2", "new_data4_wv2", "new_data5_wv2",
                       "new_data6", "new_data7", "new_data8", "new_data9",
                       "new_data_OrigScale3", "new_data_OrigScale4"
                       ]
        self.img_scale = img_scale
        self.file_lists = file_lists
        print(f"loading SingleDataset: {file_lists} with {img_scale}")
        self.file_nums = len(file_lists)
        self.dataset = {}
        self.dataset_name = dataset_name

        if 'hp' not in dataset_name:
            self.dataset = load_dataset_singlemat
        elif 'hp' in dataset_name:
            self.dataset = load_dataset_singlemat_hp
        else:
            print(f"{dataset_name} is not supported in evaluation")
            raise NotImplementedError

        if len(file_lists) == 1:
            self.test_lms, self.test_mms, self.test_ms, self.test_pan, self.gt = self.dataset(file_lists[0], self.img_scale)
            print(f"lms: {self.test_lms.shape}, ms: {self.test_ms.shape}, pan: {self.test_pan.shape}, gt: {self.gt.shape}")

    def __getitem__(self, idx):
        file_path = self.file_lists[idx % self.file_nums]
        if self.file_lists != 0:
            self.test_lms, self.test_mms, self.test_ms, self.test_pan, self.gt = self.dataset(file_path,
                                                                                              self.img_scale)

        if 'hp' not in self.dataset_name:
            return {'gt': self.gt,
                    'lms': self.test_lms,
                    'mms': self.test_mms,
                    'ms': self.test_ms,
                    'pan': self.test_pan.unsqueeze(dim=0),
                    'filename': file_path}
        else:
            return {'gt': self.gt,
                    'lms': self.test_lms,
                    'mms_hp': self.test_mms,
                    'ms_hp': self.test_ms,
                    'pan_hp': self.test_pan.unsqueeze(dim=0),
                    'filename': file_path}

    def __len__(self):
        return self.file_nums

class SingleDatasetV2(Dataset):



    def __init__(self, file_lists, dataset_name, img_scale, dataset=None):
        if dataset is None:
            dataset = ["new_data10", "new_data11", "new_data12_512",
                       "new_data3_wv2", "new_data4_wv2", "new_data5_wv2",
                       "new_data6", "new_data7", "new_data8", "new_data9",
                       "new_data_OrigScale3", "new_data_OrigScale4"
                       ]
        self.file_lists = file_lists
        self.img_scale = img_scale
        print(f"loading SingleDataset: {file_lists} with {self.img_scale}")
        self.file_nums = len(file_lists)
        self.dataset = {}
        self.dataset_name = dataset_name

        if 'hp' not in dataset_name:
            self.dataset = load_dataset_singlemat
        elif 'hp' in dataset_name:
            self.dataset = load_dataset_singlemat_hp
        else:
            print(f"{dataset_name} is not supported in evaluation")
            raise NotImplementedError

    def __getitem__(self, idx):
        file_path = self.file_lists[idx % self.file_nums]
        test_lms, test_mms, test_ms, test_pan, gt = self.dataset(file_path, self.img_scale)

        if 'hp' not in self.dataset_name:
            return {'gt': gt,
                    'lms': test_lms,
                    'mms': test_mms,
                    'ms': test_ms,
                    'pan': test_pan.unsqueeze(dim=0),
                    'filename': file_path}
        else:
            return {'gt': gt,
                    'lms': test_lms,
                    'mms_hp': test_mms,
                    'ms_hp': test_ms,
                    'pan_hp': test_pan.unsqueeze(dim=0),
                    'filename': file_path}

    def __len__(self):
        return self.file_nums


def mpl_save_fig(filename):
    plt.savefig(f"{filename}", format='svg', dpi=300, pad_inches=0, bbox_inches='tight')


def save_results(idx, save_model_output, filename, save_fmt, output):
    if filename is None:
        save_name = os.path.join(f"{save_model_output}",
                                 "output_mulExm_{}.mat".format(idx))
        sio.savemat(save_name, {'sr': output.cpu().detach().numpy()})
    else:
        filename = os.path.basename(filename).split('.')[0]
        if save_fmt != 'mat':
            output = showimage8(output)
            filename = '/'.join([save_model_output, filename + ".png"])
            # plt.imsave(filename, output, dpi=300)
            show_region_images(output, xywh=[50, 100, 50, 50], #sub_width="20%", sub_height="20%",
                               sub_ax_anchor=(0, 0, 1, 1))
            mpl_save_fig(filename)
        else:
            filename = '/'.join([save_model_output, "output_" + filename + ".mat"])
            sio.savemat(filename, {'sr': output.cpu().detach().numpy()})



