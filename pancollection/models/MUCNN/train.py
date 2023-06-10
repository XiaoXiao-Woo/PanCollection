import os
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model.Model import Unet, summaries
from lib.data import *
import time

from lib.evaluate import compute_index


import numpy as np

# ================== Pre-Define =================== #

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True

# ============= HYPER PARAMS(Pre-Defined) ==========#
lr = 0.001
epochs = 1200
ckpt = 50
batch_size = 32
model = Unet(1, 8).cuda()
summaries(model, grad=True)
model_path = "Weights/.pth"


if os.path.isfile(model_path):
    # Load the pretrained Encoder
    model.load_state_dict(torch.load(model_path))
    print('PANnet is Successfully Loaded from %s' % (model_path))



criterion = nn.L1Loss().cuda()

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999))   # optimizer 1

#milestones=[220,300,380,460]
#lr_scheduler =torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.5, last_epoch=-1)
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-7)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=200, gamma=0.1) # lr = lr* 1/gamma for each step_size = 180


def load_set(file_path):
    data = sio.loadmat(file_path)  # HxWxC

    # tensor type:
    lms = torch.from_numpy(data['lms'] / 2047).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms = torch.from_numpy(data['ms'] / 2047).permute(2, 0, 1)  # CxHxW= 8x64x64
    pan = torch.from_numpy(data['pan'] / 2047)   # HxW = 256x256
    return lms, ms, pan
    
def load_gt_compared(file_path_gt,file_path_compared):
    data1 = sio.loadmat(file_path_gt)  # HxWxC
    data2 = sio.loadmat(file_path_compared)
    gt = torch.from_numpy(data1['gt']/2047)
    compared_data = torch.from_numpy(data2['output_dmdnet_newdata6']*2047)
    return gt, compared_data


def save_checkpoint(model, epoch):  # save model function
    model_out_path = 'Weight/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)


###################################################################
# ------------------- Main Train ----------------------------------
###################################################################


def train(training_data_loader, validate_data_loader):
    others_SAM, others_ERGAS=compute_index(test_gt, test_compared_result, 4)
    print('Start training...')

    time_count = time.time()

    for epoch in range(epochs):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============lambda=============== #
        lamda1 = 0.2
        lamda2 = 0.3
        lamda3 = 0.5
        # ============Epoch Train=============== #
        model.train()
        for iteration, batch in enumerate(training_data_loader, 1):
            gt, lms, ms, pan = Variable(batch[0], requires_grad=False).cuda(), \
                               Variable(batch[1]).cuda(), \
                               Variable(batch[2]).cuda(), \
                               Variable(batch[3]).cuda()


            optimizer.zero_grad() # fixed
            gt_down1 = F.interpolate(gt, scale_factor=0.25).cuda()
            gt_down2 = F.interpolate(gt, scale_factor=0.5).cuda()
            out1,out2,out3= model(ms.float(), pan.float()) # call model

            loss1 = criterion(out1.float(), gt_down1.float())
            loss2 = criterion(out2.float(), gt_down2.float())
            loss3 = criterion(out3.float(), gt.float())
            loss = lamda1 * loss1 + lamda2 * loss2 + lamda3 * loss3 # compute loss

            epoch_train_loss.append(loss.item()) # save all losses into a vector for one epoch

            loss.backward() # fixed

            optimizer.step() # fixed

        lr_scheduler.step()  # update lr

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        # Loss0 = np.array(epoch_train_loss)
        # epoch_train_loss.append(t_loss.item())
        # np.save('/home/office-401-2/Desktop/Machine Learning/Tian-Jing Zhang/Dataset_ZHANG/BDPN_MRA/loss/epoch

        # file = open('train_loss.txt', 'a')  # write the training error into train_mse.txt
        # file.write(str(t_loss))
        # file.write('\t')
        # file.close()
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss)) # print loss for each epoch


        # ============Save model and test =============== #
        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
                save_checkpoint(model, epoch)
        if epoch % 5 == 0:  # if each ckpt epochs, then start to save model

            model.eval()
            with torch.no_grad():
                    out1, out2, out3 = model(test_ms, test_pan)
                    result_our = torch.squeeze(out3).permute(1, 2, 0)
                    result_our = result_our * 2047
                    our_SAM, our_ERGAS = compute_index(test_gt, result_our, 4)
                    print('our_SAM: {} dmdnet_SAM: {}'.format(our_SAM, others_SAM))  # print loss for each epoch
                    print('our_ERGAS: {} dmdnet_ERGAS: {}'.format(our_ERGAS, others_ERGAS))
                    print(time.time() - time_count)
        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, ms, pan = Variable(batch[0], requires_grad=False).cuda(), \
                                   Variable(batch[1]).cuda(), \
                                   Variable(batch[2]).cuda(), \
                                   Variable(batch[3]).cuda()

                gt_down1 = F.interpolate(gt, scale_factor=0.25).cuda()
                gt_down2 = F.interpolate(gt, scale_factor=0.5).cuda()

                out1, out2, out3 = model(ms.float(), pan.float())  # call model

                loss1 = criterion(out1.float(), gt_down1.float())
                loss2 = criterion(out2.float(), gt_down2.float())
                loss3 = criterion(out3.float(), gt.float())
                loss = lamda1 * loss1 + lamda2 * loss2 + lamda3 * loss3  # compute loss
                epoch_val_loss.append(loss.item())

        v_loss = np.nanmean(np.array(epoch_val_loss))
        #writer.add_scalar('val/loss', v_loss, epoch)
        # file = open('validation_loss_withoutpb.txt', 'a')  # write the training error into train_mse.txt
        # file.write(str(v_loss))
        # file.write('\t')
        # file.close()

        print('             validate loss: {:.7f}'.format(v_loss))



###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    train_set = DatasetFromHdf5('train.h5') # creat data for training
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True) # put training data to DataLoader for batches
    validate_set = DatasetFromHdf5('validation.h5') # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True) # put training data to DataLoader for batches
    # ###################################################################
    # # ------------------- load_test ----------------------------------#
    file_path = "one_image_test_file.mat"
    file_path_compared = "one_image_test_file_in_other_method.mat"
    lms, test_ms, test_pan = load_set(file_path)
    test_ms = test_ms.cuda().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
    test_pan = test_pan.cuda().unsqueeze(dim=0).unsqueeze(dim=1).float()  # convert to tensor type: 1x1xHxW
    test_gt, test_compared_result = load_gt_compared(file_path, file_path_compared)##compared_result
    test_gt = (test_gt*2047).cuda()
    test_compared_result = test_compared_result.cuda()
    ###################################################################
    time1 = time.time()
    print(time1)

    train(training_data_loader, validate_data_loader)  # call train function (call: Line 53)
    # train(training_data_loader, validate_data_loader, test_ms, test_pan, test_gt, test_compared_result)  # call train function (call: Line 53)
    print(time.time() - time1)
