import os
import h5py
import torch
import numpy as np
import scipy.io as sio
from model import ADKNet


def load_set(file_path):
    data = h5py.File(file_path)
    lrms = data["ms"][...]
    lrms = np.array(lrms, dtype=np.float32) / 2047.
    lrms = torch.from_numpy(lrms).permute(0, 3, 1, 2)

    pan = data['pan'][...]
    pan = np.array(pan, dtype=np.float32) / 2047.
    pan = np.expand_dims(pan, axis=3)
    pan = torch.from_numpy(pan).permute(0, 3, 1, 2)
    return lrms, pan


ckpt = "Weights/ADKNet4pansharpening.pth"


def test(file_path):
    lrms, pan = load_set(file_path)
    model = ADKNet().cuda().eval()
    weight = torch.load(ckpt)
    model.load_state_dict(weight)

    with torch.no_grad():
        for i in range(lrms.shape[0]):
            x1, x2 = lrms[i, :, :, :], pan[i, :, :, :]

            x1 = x1.cuda().unsqueeze(dim=0).float()
            x2 = x2.cuda().unsqueeze(dim=0).float()
            sr = model(x1, x2)
            sr = torch.clamp(sr, 0, 1)
            sr = torch.squeeze(sr).permute(1, 2, 0).cpu().detach().numpy()

            save_name = os.path.join("results/WV3_reduced_resolution_multi_exm_ADKNet", "{}-test.mat".format(i))
            #save_name = os.path.join("results/WV3_full_resolution_multi_exm_ADKNet", "{}-test.mat".format(i))

            sio.savemat(save_name, {'sr': sr})


if __name__ == '__main__':
    file_path = "test_data/test1_mulExm1258.mat"
    #file_path = "test_data/test1_mulExm_OrigScale.mat"
    test(file_path)
