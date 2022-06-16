import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import Dataset_Pro
import scipy.io as sio
from model import LACNET
import numpy as np
import shutil
from torch.utils.tensorboard import SummaryWriter



# ================== Pre-test =================== #
def load_set(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    # tensor type:
    lms = torch.from_numpy(data['lms'] / 2047.0).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms = torch.from_numpy((data['ms'] / 2047.0)).permute(2, 0, 1)  # CxHxW= 8x64x64
    pan = torch.from_numpy((data['pan'] / 2047.0))   # HxW = 256x256

    return lms, ms, pan

def load_gt_compared(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    # tensor type:
    test_gt = torch.from_numpy(data['gt'] / 2047.0)  # CxHxW = 8x256x256

    return test_gt






# ================== Pre-Define =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True

# ============= HYPER PARAMS(Pre-Defined) ==========#
lr = 0.0003 # 0.001
epochs = 500
ckpt = 50
batch_size = 32
device=torch.device('cuda:1')

model = LACNET().to(device)
#model.load_state_dict(torch.load('/Data/Machine Learning/Zi-Rong Jin/pan/o/DKNET_500.pth'))
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999))   # optimizer 1


if os.path.exists('train_logs'):  # for tensorboard: copy dir of train_logs
    shutil.rmtree('train_logs')  # ---> console (see tensorboard): tensorboard --logdir = dir of train_logs
writer = SummaryWriter('train_logs')


def save_checkpoint(model, epoch):  # save model function
    model_out_path = 'DKNET.pth'
    #if not os.path.exists(model_out_path):
    #    os.makedirs(model_out_path)
    torch.save(model.state_dict(), model_out_path)


###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################

def train(training_data_loader, validate_data_loader,test_lms,test_ms, test_pan, test_gt):
    print('Start training...')

    for epoch in range(epochs):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            gt, lms, _, _, pan = Variable(batch[0], requires_grad=False).to(device), \
                                     Variable(batch[1]).to(device), \
                                     batch[2], \
                                     batch[3], \
                                     Variable(batch[4]).to(device)
            optimizer.zero_grad()  # fixed
            out = model(pan, lms)

            loss = criterion(out, gt)  # compute loss
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch


            loss.backward()  # fixed
            optimizer.step()  # fixed

 #       lr_scheduler.step()  # update lr

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        writer.add_scalar('train/loss', t_loss, epoch)  # write to tensorboard to check
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch

        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)
        # if epoch % 10 == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         output1, output2, output3 = model(test_ms, test_pan,test_lms)
        #         result_our = torch.squeeze(output3).permute(1, 2, 0)
        #         #sr = torch.squeeze(output3).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
        #         result_our = result_our * 2047
        #         result_our = result_our.type(torch.DoubleTensor).to(device)
        #
        #         our_SAM, our_ERGAS = compute_index(test_gt, result_our, 4)
        #         print('our_SAM: {} dmdnet_SAM: 2.9355'.format(our_SAM) ) # print loss for each epoch
        #         print('our_ERGAS: {} dmdnet_ERGAS:1.8119 '.format(our_ERGAS))
        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, _, _ ,pan= Variable(batch[0], requires_grad=False).to(device), \
                                         Variable(batch[1]).to(device), \
                                         batch[2], \
                                         batch[3], \
                                         Variable(batch[4]).to(device)

                out = model(pan,lms)


                loss = criterion(out, gt)
                epoch_val_loss.append(loss.item())



        v_loss = np.nanmean(np.array(epoch_val_loss))
        writer.add_scalar('val/loss', v_loss, epoch)
        print('validate loss: {:.7f}'.format(v_loss))

    writer.close()  # close tensorboard


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":

    train_set = Dataset_Pro('/Data/Machine Learning/Zi-Rong Jin/jinzirong/training_data/train.h5')  # creat data for training
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = Dataset_Pro('/Data/Machine Learning/Zi-Rong Jin/jinzirong/training_data/valid.h5')  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches
    # ------------------- load_test ----------------------------------#
    file_path = "test_data/new_data6.mat"
    test_lms, test_ms, test_pan = load_set(file_path)
    test_lms = test_lms.to(device).unsqueeze(dim=0).float()
    test_ms = test_ms.to(device).unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
    test_pan = test_pan.to(device).unsqueeze(dim=0).unsqueeze(dim=1).float()  # convert to tensor type: 1x1xHxW
    test_gt= load_gt_compared(file_path)  ##compared_result
    test_gt = (test_gt * 2047).to(device).double()
    ###################################################################
    #train(training_data_loader, validate_data_loader)  # call train function (call: Line 53)
    train(training_data_loader, validate_data_loader,test_lms,test_ms, test_pan, test_gt)  # call train function (call: Line 66)
