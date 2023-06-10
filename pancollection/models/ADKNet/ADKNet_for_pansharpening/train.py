import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from data import Dataset_Pro
from model import ADKNet, summaries
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = True
cudnn.deterministic = True
cudnn.benchmark = False

lr = 0.003
ckpt = 10
epochs = 1000
start_epoch = 0
batch_size = 32

model = ADKNet().cuda()
summaries(model, grad=True)

criterion = nn.MSELoss(size_average=True).cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.5)


def save_checkpoint(model, epoch):
    model_out_path = 'Weights' + '/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)


def train(training_data_loader, validate_data_loader, start_epoch=0):
    t1 = time.time()
    print('Start training...')

    for epoch in range(start_epoch, epochs, 1):
        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            gt, lrms, pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            optimizer.zero_grad()
            sr = model(lrms, pan)
            loss = criterion(sr, gt)
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        t_loss = np.nanmean(np.array(epoch_train_loss))
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))

        if epoch % ckpt == 0:
            save_checkpoint(model, epoch)


        with torch.no_grad():
            model.eval()
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lrms, pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
                sr = model(lrms, pan)
                loss = criterion(sr, gt)
                epoch_val_loss.append(loss.item())

        if epoch % 10 == 0:
            v_loss = np.nanmean(np.array(epoch_val_loss))
            print('      validate loss: {:.7f}'.format(v_loss))
            t2 = time.time()
            print('        time cost: {:.4f}s'.format(t2 - t1))


if __name__ == "__main__":
    train_set = Dataset_Pro('./training_data/train.h5')
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
    validate_set = Dataset_Pro('./training_data/valid.h5')
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
    train(training_data_loader, validate_data_loader, start_epoch)

