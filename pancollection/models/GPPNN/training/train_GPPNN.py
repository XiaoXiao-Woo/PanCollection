# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

import os
import xlwt
import time
import datetime

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader 
from scipy.io import savemat

import sys
sys.path.append("..")
from models import get_sat_param
from models.GPPNN import GPPNN
from metrics import get_metrics_reduced
from utils import PSH5Dataset, PSDataset, prepare_data, normlization, save_param, psnr_loss, ssim


'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''



model_str = 'GPPNN'
satellite_str = 'Landsat8'


# . Get the parameters of your satellite
sat_param = get_sat_param(satellite_str)
if sat_param!=None:
    ms_channels, pan_channels, scale = sat_param
else:
    print('You should specify `ms_channels`, `pan_channels` and `scale`! ')
    ms_channels = 10
    pan_channels = 1
    scale = 2


# . Set the hyper-parameters for training
num_epochs = 100
lr = 5e-4
weight_decay = 0
batch_size = 16
n_layer = 8
n_feat = 64

# . Get your model 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = GPPNN(ms_channels,
           pan_channels, 
           n_feat,
           n_layer).to(device)
print(net)


# . Get your optimizer, scheduler and loss function
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
loss_fn = nn.L1Loss().to(device)

# . Create your data loaders
prepare_data_flag = False # set it to False, if you have prepared dataset
train_path      = '../PS_data/%s/%s_train.h5'%(satellite_str,satellite_str)
validation_path = '../PS_data/%s/validation'%(satellite_str)
test_path       = '../PS_data/%s/test'%(satellite_str)
if prepare_data_flag is True:
    prepare_data(data_path = '../PS_data/%s'%(satellite_str), 
                 patch_size=32, aug_times=1, stride=32, synthetic=False, scale=scale,
                 file_name = train_path)

trainloader      = DataLoader(PSH5Dataset(train_path), 
                              batch_size=batch_size, 
                              shuffle=True) #[N,C,K,H,W]
validationloader = DataLoader(PSDataset(validation_path,scale),      
                              batch_size=1)
testloader = DataLoader(PSDataset(test_path, scale),
                        batch_size=1)
loader = {'train':      trainloader,
          'validation': validationloader}


# . Creat logger
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
save_path = os.path.join(
    'logs/%s'%(model_str),
    timestamp+'_%s_layer%d_filter_%d'%(satellite_str,n_layer,n_feat)
    )
writer = SummaryWriter(save_path)
params = {'model': model_str,
          'satellite': satellite_str,
          'epoch': num_epochs,
          'lr': lr,
          'batch_size': batch_size,
          'n_feat': n_feat,
          'n_layer': n_layer}
save_param(params,
           os.path.join(save_path, 'param.json'))


'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''


step = 0
best_psnr_val,psnr_val, ssim_val = 0., 0., 0.
torch.backends.cudnn.benchmark = True
prev_time = time.time()


for epoch in range(num_epochs):
    ''' train '''
    for i, (ms, pan, gt) in enumerate(loader['train']):
        # 0. preprocess data
        ms, pan, gt = ms.cuda(), pan.cuda(), gt.cuda()
        ms,_ = normlization(ms.cuda())
        pan,_ = normlization(pan.cuda())
        gt,_ = normlization(gt.cuda())
        
        # 1. update
        net.train()
        net.zero_grad()
        optimizer.zero_grad()
        imgf = net(ms, pan)
        loss = loss_fn(gt, imgf)
        loss.backward()
        optimizer.step() 
        
        # 2. print
        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] [PSNR/Best: %.4f/%.4f] ETA: %s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                psnr_val,
                best_psnr_val,
                time_left,
            )
        )

        # 3. Log the scalar values
        writer.add_scalar('loss', loss.item(), step)
        writer.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], step)
        step+=1
        

    ''' validation ''' 
    current_psnr_val = psnr_val
    psnr_val = 0.
    ssim_val = 0.
    with torch.no_grad():
        net.eval()
        for i, (ms, pan, gt) in enumerate(loader['validation']):
            ms,_ = normlization(ms.cuda())
            pan,_ = normlization(pan.cuda())
            gt,_ = normlization(gt.cuda())
            imgf = net(ms, pan)
            psnr_val += psnr_loss(imgf, gt, 1.)
            ssim_val += ssim(imgf, gt, 11, 'mean', 1.)
        psnr_val = float(psnr_val/loader['validation'].__len__())
        ssim_val = float(ssim_val/loader['validation'].__len__())
    writer.add_scalar('PSNR/val', psnr_val, epoch)
    writer.add_scalar('SSIM/val', ssim_val, epoch)
    
    psnr_val = 0.
    ssim_val = 0.
    metrics = torch.zeros(2,testloader.__len__())
    with torch.no_grad():
        net.eval()
        for i, (ms, pan, gt) in enumerate(testloader):
            ms,_ = normlization(ms.cuda())
            pan,_ = normlization(pan.cuda())
            gt,_ = normlization(gt.cuda())
            imgf = net(ms, pan)
            metrics[:,i] = torch.Tensor(get_metrics_reduced(imgf, gt))[:2]
        psnr_val, ssim_val = metrics.mean(dim=1)
    writer.add_scalar('PSNR/test', psnr_val, epoch)
    writer.add_scalar('SSIM/test', ssim_val, epoch)  
        
    ''' save model ''' 
    # Save the best weight
    if best_psnr_val<psnr_val:
        best_psnr_val = psnr_val
        torch.save({'net':net.state_dict(), 
                    'optimizer':optimizer.state_dict(),
                    'epoch':epoch},
                   os.path.join(save_path, 'best_net.pth'))
    # Save the current weight
    torch.save({'net':net.state_dict(),
                'optimizer':optimizer.state_dict(),
                'epoch':epoch},
                os.path.join(save_path, 'last_net.pth'))
    
    ''' backtracking '''
    if epoch>0:
        if torch.isnan(loss):
            print(10*'='+'Backtracking!'+10*'=')
            net.load_state_dict(torch.load(os.path.join(save_path, 'best_net.pth'))['net'])
            optimizer.load_state_dict(torch.load(os.path.join(save_path, 'best_net.pth'))['optimizer'])


'''
------------------------------------------------------------------------------
Test
------------------------------------------------------------------------------
'''


# 1. Load the best weight and create the dataloader for testing
net.load_state_dict(torch.load(os.path.join(save_path, 'best_net.pth'))['net'])



# 2. Compute the metrics
metrics = torch.zeros(5,testloader.__len__())
with torch.no_grad():
    net.eval()
    for i, (ms, pan, gt) in enumerate(testloader):
        ms,_ = normlization(ms.cuda())
        pan,_ = normlization(pan.cuda())
        gt,_ = normlization(gt.cuda())
        imgf = net(ms, pan)
        metrics[:,i] = torch.Tensor(get_metrics_reduced(imgf, gt))
        savemat(os.path.join(save_path,testloader.dataset.files[i].split('\\')[-1]),
               {'HR':imgf.squeeze().detach().cpu().numpy()} )


# 3. Write the metrics
f = xlwt.Workbook()
sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True)
img_name = [i.split('\\')[-1].replace('.mat','') for i in testloader.dataset.files]
metric_name = ['PSNR','SSIM','CC','SAM','ERGAS']
for i in range(len(metric_name)):
    sheet1.write(i+1,0,metric_name[i])
for j in range(len(img_name)):
   sheet1.write(0,j+1,img_name[j])  
for i in range(len(metric_name)):
    for j in range(len(img_name)):
        sheet1.write(i+1,j+1,float(metrics[i,j]))
sheet1.write(0,len(img_name)+1,'Mean')
for i in range(len(metric_name)):
    sheet1.write(i+1,len(img_name)+1,float(metrics.mean(1)[i]))
f.save(os.path.join(save_path,'test_result.xls'))