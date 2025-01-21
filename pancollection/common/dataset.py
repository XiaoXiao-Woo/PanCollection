import imageio
import torch
from torch.utils.data import Dataset
import h5py
import cv2
import numpy as np
import os
import torch.nn.functional as F
import logging
# from scipy import io as sio
from udl_vis.Basis.postprocess import showimage8
import matplotlib.pyplot as plt
from scipy import io as sio

logger = logging.getLogger(__name__)
class Dataset_Pro(Dataset):
    def __init__(self, file_path, img_scale):
        super(Dataset_Pro, self).__init__()

        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3
        print(f"loading Dataset_Pro: {file_path} with {img_scale}, keys: {data.keys()}")
        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        num = len(gt1)
        gt1 = np.array(gt1, dtype=np.float32) / img_scale
        self.gt = torch.from_numpy(gt1)[:num, ...]  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / img_scale

        self.ms = torch.from_numpy(ms1)[:num, ...]

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / img_scale
        self.lms = torch.from_numpy(lms1)[:num, ...]

        pan1 = data["pan"][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / img_scale  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)[:num, ...]  # Nx1xHxW:
        print(
            f"gt: {self.gt.size()}, lms: {self.lms.size()}, pan: {self.pan.size()}, ms: {self.ms.size()} with {img_scale}"
        )

    #####必要函数
    def __getitem__(self, index):
        # print(os.environ.get("RANK", -1), index)
        return {
            "gt": self.gt[index, :, :, :].float(),
            "lms": self.lms[index, :, :, :].float(),
            "ms": self.ms[index, :, :, :].float(),
            "pan": self.pan[index, :, :, :].float(),
        }

        #####必要函数

    def __len__(self):
        return self.gt.shape[0]


def load_dataset_H5_hp(file_path, scale, suffix, use_cuda=True):
    if suffix == ".h5":
        data = h5py.File(file_path)  # CxHxW
        print(data.keys())
    else:
        data = sio.loadmat(file_path)
    # shape_list = []
    # for k in data.keys():
    #     shape_list.append((k, data[k].shape))
    # print(shape_list)

    # tensor type: NxCxHxW:

    lms = torch.from_numpy(data["lms"][...] / scale).float()  # .permute(0, 3, 1, 2)
    ms = data["ms"][...] / scale
    ms_hp = torch.from_numpy(get_edge(ms)).float()  # .permute(0, 3, 1, 2)  # NxCxHxW:
    ms = torch.from_numpy(ms).float()
    # mms_hp = torch.nn.functional.interpolate(ms_hp, size=(ms_hp.size(2) * 2, ms_hp.size(3) * 2),
    #                                       mode="bilinear", align_corners=True)
    pan = np.squeeze(data["pan"][...])
    pan = pan[:, np.newaxis, :, :]  # NxCxHxW (C=1)
    pan_hp = torch.from_numpy(
        get_edge(pan / scale)
    ).float()  # .permute(0, 3, 1, 2)  # Nx1xHxW:
    pan = torch.from_numpy(pan / scale).float()
    if data.get("gt", None) is None:
        gt = torch.from_numpy(data["lms"][...] / scale).float()
    else:
        gt = torch.from_numpy(data["gt"][...] / scale).float()

    return {
        "lms": lms,
        # 'mms:': mms_hp,
        "ms": ms,
        "ms_hp": ms_hp,
        "pan_hp": pan_hp,
        "pan": pan,
        "gt": gt,
    }


def load_dataset_H5(file_path, scale, suffix, use_cuda=True):
    if suffix == ".h5":
        data = h5py.File(file_path)  # CxHxW
        print(data.keys())
    else:
        data = sio.loadmat(r"D:\Datasets\wzc\E1_r_gf2_82.mat")

    # tensor type:
    if use_cuda:
        lms = (
            torch.from_numpy(data["lms"][...] / scale).cuda().float()
        )  # CxHxW = 8x64x64

        ms = torch.from_numpy(data["ms"][...] / scale).cuda().float()  # CxHxW= 8x64x64
        mms = torch.nn.functional.interpolate(
            ms,
            size=(ms.size(2) * 2, ms.size(3) * 2),
            mode="bilinear",
            align_corners=True,
        )
        pan = torch.from_numpy(data["pan"][...] / scale).cuda().float()  # HxW = 256x256

        gt = torch.from_numpy(data["gt"][...] / scale).cuda().float()

    else:
        lms = torch.from_numpy(data["lms"][...] / scale).float()  # CxHxW = 8x64x64

        ms = torch.from_numpy(data["ms"][...] / scale).float()  # CxHxW= 8x64x64
        mms = torch.nn.functional.interpolate(
            ms,
            size=(ms.size(2) * 2, ms.size(3) * 2),
            mode="bilinear",
            align_corners=True,
        )
        pan = torch.from_numpy(data["pan"][...] / scale).float()  # HxW = 256x256

        if data.get("gt", None) is None:
            gt = torch.from_numpy(data["lms"][...] / scale).float()
        else:
            if np.max(data["gt"][...]) > 1:
                gt = torch.from_numpy(data["gt"][...] / scale).float()
            else:
                gt = torch.from_numpy(data["gt"][...]).float()
    num = 20
    return {
        "lms": lms[:num],
        "mms:": mms[:num],
        "ms": ms[:num],
        "pan": pan[:num],
        "gt": gt[:num],
    }


class MultiExmTest_h5(Dataset):

    def __init__(self, file_path, dataset_name, img_scale, suffix=".h5"):
        super(MultiExmTest_h5, self).__init__()

        # self.scale = 2047.0
        # if 'gf' in dataset_name:
        #     self.scale = 1023.0
        self.img_scale = img_scale
        self.dataset_name = dataset_name
        print(f"loading MultiExmTest_h5: {file_path} with {img_scale}")
        # 一次性载入到内存
        if "hp" not in dataset_name:
            data = load_dataset_H5(file_path, img_scale, suffix, False)

        elif "hp" in dataset_name:
            file_path = file_path.replace("_hp", "")
            data = load_dataset_H5_hp(file_path, img_scale, suffix, False)

        else:
            print(f"{dataset_name} is not supported in evaluation")
            raise NotImplementedError

        if file_path.endswith("mat"):
            self.lms = data["lms"].permute(0, 3, 1, 2)  # CxHxW = 8x256x256
            self.ms = data["ms"].permute(0, 3, 1, 2)  # CxHxW= 8x64x64
            # self.mms = torch.nn.functional.interpolate(self.ms, size=(self.ms.size(2) * 2, self.ms.size(3) * 2),
            #                                            mode="bilinear", align_corners=True)
            self.pan = data["pan"]
            self.gt = data["gt"].permute(0, 3, 1, 2)
            if "hp" in dataset_name:
                self.ms_hp = data["ms_hp"].permute(0, 3, 1, 2)
                self.pan_hp = data["pan_hp"]

        elif file_path.endswith("h5"):
            self.lms = data["lms"]
            if "hp" in dataset_name:
                self.ms_hp = data["ms_hp"]
                self.pan_hp = data["pan_hp"]
            # if 'mms' in data.keys():
            #     self.mms = torch.nn.functional.interpolate(self.ms, size=(self.ms.size(2) * 2, self.ms.size(3) * 2),
            #                                            mode="bilinear", align_corners=True)
            self.pan = data["pan"]
            self.gt = data["gt"]  # .permute([0, 2, 3, 1])
            self.ms = data["ms"]

        print(
            f"lms: {self.lms.shape}, ms: {self.ms.shape}, pan: {self.pan.shape}, gt: {self.gt.shape}"
        )

    def __getitem__(self, item):

        if "hp" in self.dataset_name:
            return {
                "lms": self.lms[item, ...],
                # 'mms': self.mms[item, ...],
                "ms": self.ms[item, ...],
                "ms_hp": self.ms_hp[item, ...],
                "pan_hp": self.pan_hp[item, ...],
                "pan": self.pan[item, ...],
                "gt": self.gt[item, ...],
            }
        else:
            return {
                "lms": self.lms[item, ...],
                # 'mms': self.mms[item, ...],
                "ms": self.ms[item, ...],
                "pan": self.pan[item, ...],
                "gt": self.gt[item, ...],
            }

    def __len__(self):
        return self.gt.shape[0]


def mpl_save_fig(filename):
    plt.savefig(f"{filename}", format="svg", dpi=300, pad_inches=0, bbox_inches="tight")


def get_save_model_name(save_model_output, filename, idx, fmt):
    if filename is None:
        fname = os.path.join(
            f"{save_model_output}", f"output_mulExm_{idx}.{fmt}"
        )
    else:
        if fmt != "mat":
            fname = os.path.basename(filename + f"_{idx}").split(".")[0]
            fname = "/".join([save_model_output, filename + ".png"])
        else:
            fname = "/".join([save_model_output, "output_" + filename + ".mat"])
    return fname


def save_results(idx, save_model_output, filename, save_fmt, output, img_scale):

    if output.shape[-1] == 4:
        channels = [0, 1, 2]
    else:
        channels = [0, 2, 4]

    for fmt in save_fmt:
        for i in range(output.shape[0]):
            fname = get_save_model_name(
                save_model_output, filename, idx * output.shape[0] + i, fmt
            )
            if fmt != "mat":
                try:
                    out = showimage8(output[i], unnormlize=img_scale, channels=channels)
                except Exception as e:
                    logger.error(e)
                    out = output[..., channels].cpu().detach().numpy()
                if out.max() <= 1:
                    out = np.ceil(out * 255).astype(np.uint8)
                else:
                    out = out.astype(np.uint8)
                imageio.imwrite(fname, out)
            else:
                # filename = '/'.join([save_model_output, "output_" + filename + ".mat"])
                sio.savemat(fname, {"sr": output[i].cpu().detach().numpy()})

            # TODO: another file to implement the result visualization
            # filename = '/'.join([save_model_output, filename + ".png"])
            # plt.imsave(filename, output, dpi=300)
            # show_region_images(output, xywh=[50, 100, 50, 50],  # sub_width="20%", sub_height="20%",
            #                    sub_ax_anchor=(0, 0, 1, 1))
            # mpl_save_fig(filename)


if __name__ == "__main__":

    file_path = (
        "/Data/Datasets/pansharpening_2/PanCollection/test_data/test_wv3_multiExm1.h5"
    )
    # file_path = "/Data/Datasets/pansharpening_2/PanCollection/test_data/WV3/RR-Data/Test(HxWxC)_wv3_data10.mat"
    scale = 2047.0
    data = h5py.File(file_path)  # HxWxC
    print(f"loading Dataset_Pro: {file_path}, keys: {data.keys()}")
    # tensor type:
    lms = torch.from_numpy(data["lms"][...] / scale).permute(
        2, 0, 1
    )  # B, H, W, C: 0, 3, 1, 2 / H,W,C: 2, 0, 1
    ms = (
        torch.from_numpy(data["ms"][...] / scale).permute(2, 0, 1).unsqueeze(dim=0)
    )  # CxHxW= 8x64x64
    mms = F.interpolate(
        ms, size=(ms.size(2) * 2, ms.size(3) * 2), mode="bilinear", align_corners=True
    )
    pan = torch.from_numpy(data["pan"][...] / scale)  # HxW = 256x256
    if data.get("gt", None) is None:
        gt = torch.from_numpy(data["lms"][...] / scale)
    else:
        gt = torch.from_numpy(data["gt"][...] / scale)
