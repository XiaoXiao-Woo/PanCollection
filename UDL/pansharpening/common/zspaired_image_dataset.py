from torch.utils import data as data
from torchvision.transforms.functional import normalize
from PIL import Image
import numpy as np
import random
import cv2
import glob
import scipy.io as sio
import torch
from torch.nn import functional as F


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            # cv2.flip(img, 1, img)
            img = img[:, :, ::-1, :]
        if vflip:  # vertical
            # cv2.flip(img, 0, img)
            img = img[:, :, :, ::-1]
        if rot90:
            img = img.transpose(-1, -2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs

class ZSPairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt, data_path):
        super(ZSPairedImageDataset, self).__init__()
        self.opt = opt
        self.paths = [data_path]
        # self.paths = glob.glob(data_dir + ".mat")
        # print(self.paths, data_dir)


    def __getitem__(self, index):

        scale = self.opt.scale

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        # path = self.paths[index]

        data = sio.loadmat(self.paths[0])
        pan, hs = data['hs'], data['pan']
        pan = torch.from_numpy(pan / 65535.0).view(1, 1, *pan.shape).float()
        hs = torch.from_numpy(hs / 65535.0).view(1, *hs.shape).float()

        pan, hs = paired_random_crop(pan, hs, self.opt.patch_size, 6)


        # augmentation for training
        if not self.opt.eval:
            # gt_size = self.opt['gt_size']
            # random crop
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # Zero-shot construction
            # Case1 从GT里构造HR, LR，ZSSR是先让输入数据维度相同再做超分
            # 原始数据要作为验证集的

            # 因此，仿真数据: 原始数据进行下采样再上采样和MTF
            pan_lq = F.interpolate(pan, (int(pan.shape[-1] / scale),
                                   int(pan.shape[-1] / scale)),
                                   mode="bicubic")
            # pan = F.interpolate(pan_lq, (int(pan_lq.shape[-1] * scale), \
            #                     int(pan_lq.shape[-1] * scale)),
            #                     mode="bicubic")


            hs_lq = F.interpolate(hs, (int(hs.shape[-1] / scale),
                                 int(hs.shape[-1] / scale)),
                                  mode="bicubic"
                                  )
            hs_lq = F.interpolate(hs_lq, (int(hs_lq.shape[-1] * scale), \
                                 int(hs_lq.shape[-1] * scale)),
                               mode="bicubic")

            # Case2 从GT里构造HR, 保留原来的LR
            # pan, pan_lq = np.array(pan) , np.array(pan_lq) / 65535.0
            # hs, hs_lq = np.array(hs) / 65535.0, np.array(hs_lq) / 65535.0
            # flip, rotation
            # pan_gt, pan_lq = augment([pan, pan_lq], hflip=True, rotation=False)
            # hs_gt, hs_lq = augment([hs, hs_lq], hflip=True, rotation=False)

            # BGR to RGB, HWC to CHW, numpy to tensor
            # pan, pan_lq, hs, hs_lq = img2tensor([pan, pan_lq, hs, hs_lq], bgr2rgb=False, float32=True)
            # print(f"{self.paths}, pan_gt: {pan.shape}, hs_gt: {hs.shape}, pan_lq:{pan_lq.shape}, hs_lq: {hs_lq.shape}")
            return {'pan_gt': pan[0], 'hs_gt': hs[0], 'pan_lq': pan_lq[0], 'hs_lq': hs_lq[0]}
        else:
            # BGR to RGB, HWC to CHW, numpy to tensor
            pan, hs = img2tensor([pan, hs], bgr2rgb=False, float32=True)

            return {'pan': pan, 'hs': hs}

    def __len__(self):
        return len(self.paths)
