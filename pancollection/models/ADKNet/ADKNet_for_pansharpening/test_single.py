import os
import torch
import scipy.io as sio
from model import ADKNet


def load_set(file_path):
    data = sio.loadmat(file_path)
    ms = torch.from_numpy(data['ms'] / 2047.0)
    ms = ms.numpy()
    lrms = torch.from_numpy(ms).permute(2, 0, 1)
    pan = torch.from_numpy(data['pan'] / 2047.0)

    return lrms, pan


ckpt = "Weights/ADKNet4pansharpening.pth"


def test(file_path):
    lrms, pan = load_set(file_path)
    model = ADKNet().cuda().eval()
    weight = torch.load(ckpt)
    model.load_state_dict(weight)

    with torch.no_grad():

        x1, x2 = lrms, pan
        x1 = x1.cuda().unsqueeze(dim=0).float()
        x2 = x2.cuda().unsqueeze(dim=0).unsqueeze(dim=1).float()
        sr = model(x1, x2)
        sr = torch.clamp(sr, 0, 1)
        sr = torch.squeeze(sr).permute(1, 2, 0).cpu().detach().numpy()

        save_name = os.path.join("results", "new_data5_wv2_ADKNet.mat")
        #save_name = os.path.join("results", "new_data6_ADKNet.mat")
        #save_name = os.path.join("results", "new_data7_ADKNet.mat")

        sio.savemat(save_name, {'new_data5_wv2_ADKNet': sr})
        #sio.savemat(save_name, {'new_data6_ADKNet': sr})
        #sio.savemat(save_name, {'new_data7_ADKNet': sr})


if __name__ == '__main__':
    file_path = "test_data/new_data5_wv2.mat"
    #file_path = "test_data/new_data6.mat"
    #file_path = "test_data/new_data7.mat"
    test(file_path)
