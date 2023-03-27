import glob
import torch
from torch.utils.data import DataLoader
from common.test_panloader import oldPan_dataloader, PanCollection_dataloader, DLPan_dataloader
from common.train_panloader import oldPan_trainloader, PanCollection_trainloader, DLPan_trainloader
from common.valid_panloader import oldPan_validloader, PanCollection_validloader, DLPan_validloader

class PansharpeningSession():
    def __init__(self, args):
        self.dataloaders = {}
        self.samples_per_gpu = args.samples_per_gpu
        self.workers_per_gpu = args.workers_per_gpu
        # self.patch_size = args.patch_size
        self.writers = {}
        self.args = args

        # self.mapping = {'wv3': 'wv3_multiExm1.h5', 'wv2': 'wv2_multiExm1.h5', 'qb': 'qb_multiExm1.h5', 'gf2': 'gf2_multiExm1.h5'}

    def get_dataloader(self, dataset_name, distributed):

        dataset = None
        dataloader_name = self.args.dataloader_name
        if dataloader_name == "oldPan_dataloader":
            dataset = oldPan_trainloader(dataset_name, self.args)
        elif dataloader_name == "DLPan_dataloader":
            dataset = DLPan_trainloader(dataset_name, self.args)
        elif dataloader_name == "PanCollection_dataloader":
            dataset = PanCollection_trainloader(dataset_name, self.args)

        if dataset is None:
            raise NotImplementedError(f"{dataset_name} or {dataloader_name} is not supported.")

        sampler = None
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        # if not dataset_name in self.dataloaders:
        dataloaders = \
            DataLoader(dataset, batch_size=self.samples_per_gpu,
                       persistent_workers=(True if self.workers_per_gpu > 0 else False), pin_memory=True,
                       shuffle=(sampler is None), num_workers=self.workers_per_gpu, drop_last=True, sampler=sampler)

        return dataloaders, sampler

    def get_valid_dataloader(self, dataset_name, distributed):

        dataset = None
        dataloader_name = self.args.dataloader_name
        if dataloader_name == "oldPan_dataloader":
            dataset = oldPan_validloader(dataset_name, self.args)
        elif dataloader_name == "DLPan_dataloader":
            dataset = DLPan_validloader(dataset_name, self.args)
        elif dataloader_name == "PanCollection_dataloader":
            dataset = PanCollection_validloader(dataset_name, self.args)

        if dataset is None:
            raise NotImplementedError(f"{dataset_name} or {dataloader_name} is not supported.")

        sampler = None
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        # if not dataset_name in self.dataloaders:
        dataloaders = \
            DataLoader(dataset, batch_size=self.samples_per_gpu,
                       persistent_workers=(True if self.workers_per_gpu > 0 else False), pin_memory=True,
                       shuffle=(sampler is None), num_workers=self.workers_per_gpu, drop_last=True, sampler=sampler)

        return dataloaders, sampler

    def get_eval_dataloader(self, dataset_name, distributed):


        dataloader_name = self.args.dataloader_name
        if dataloader_name == "oldPan_dataloader":
            dataset = oldPan_dataloader(dataset_name, self.args)
        elif dataloader_name == "DLPan_dataloader":
            dataset = DLPan_dataloader(dataset_name, self.args)
        elif dataloader_name == "PanCollection_dataloader":
            dataset = PanCollection_dataloader(dataset_name, self.args)
        else:
            dataset = None

        if dataset is None:
            raise NotImplementedError(f"{dataset_name} or {dataloader_name} is not supported.")

        sampler = None
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        # if not dataset_name in self.dataloaders:
        dataloaders = \
            DataLoader(dataset, batch_size=1,
                       shuffle=False, num_workers=1, drop_last=False, sampler=sampler)
        return dataloaders, sampler


def test_my_data(args, sess):
    # oldPan: RR: ['gt', 'lms', 'ms', 'pan'] FR: ['lms', 'ms', 'pan']
    # 12,580 = 0.7 0.2 0.1
    # train_wv3.h5: 8806, 16-64
    # valid_wv3.h5: 2516
    # test1_mulExm1258.mat: 1258, 16-64

    # test_wv3_mulExm.h5: 78, 64-256
    # test1_mulExm_OrigScale.mat: 200, 64-256

    # test_wv2_mulExm: 93, 64-256

    # train_qb.h5 20685 16-64
    # valid_qb.h5
    # test_qb_multiExm.h5 48 64-256

    # qb48.mat: 48, 64-256
    # TestData_QB: 48

    # train_gf2.h5 21607 16-64
    # valid_gf2.h5
    # test_gf2_mulExm.h5: 84, 64-256 ???

    # gf81.mat: 81, 64-256

    args.data_dir = "D:/Datasets/pansharpening/oldPan"
    # args.dataset = 'train_wv3.h5'
    # loader, _ = sess.get_dataloader(args.dataset, False)

    # args.dataset = 'valid_wv3'
    # loader, _ = sess.get_test_dataloader(args.dataset, False)

    args.dataset = 'test_gf2_mulExm_84.h5'
    loader, _ = sess.get_eval_dataloader(args.dataset, False)

    print(len(loader))

def test_PanCollection(args, sess):
    # survey: RR: ['gt', 'lms', 'ms', 'pan'] FR: ['lms', 'ms', 'pan']
    # train_wv3.h5 9714 16-64 / / 20
    # train_wv2.h5 15084 16-64
    # train_gf2.h5 19809 16-64
    # train_qb.h5  17139 16-64

    # test_wv3_OrigScale_mulExm.h5: 126, 128-512
    # test_gf2_OrigScale_mulExm.h5: 318 128-512
    # test_qb_OrigScale_multiExm.h5

    args.data_dir = "D:/Datasets/pansharpening/PanCollection"
    args.dataset = 'train_gf2.h5'
    loader, _ = sess.get_dataloader(args.dataset, False)

    # TODO: validate
    # args.dataset = 'wv3'
    # loader, _ = sess.get_test_dataloader(args.dataset, False)

    # args.dataset = 'wv3_multiExm1.h5'
    # loader, _ = sess.get_eval_dataloader(args.dataset, False)
    print(len(loader))

def test_DLPan(args, sess):
    # DLPan: RR: ['gt', 'lms', 'ms', 'pan'] FR: ['lms', 'ms', 'pan']
    # train_wv3_10000: 9000 16-64
    # valid_wv3_10000: 1000 16-64
    # TestData_wv3.h5: 4 128-512
    # Single: NY1_WV3_FR.mat, NY1_WV3_RR.mat

    # qb 9000 16-64

    args.data_dir = "D:/Datasets/pansharpening/DLPan"
    # args.dataset = 'train_qb_10000.h5'
    # loader, _ = sess.get_dataloader(args.dataset, False)

    # args.dataset = 'valid_wv3_10000.h5'
    # loader, _ = sess.get_test_dataloader(args.dataset, False)

    args.dataset = 'TestData_wv3.h5'
    # args.dataset = 'NY1_WV3_FR.mat'
    loader, _ = sess.get_eval_dataloader(args.dataset, False)

    print(len(loader))


if __name__ == '__main__':
    # from option import args
    import argparse
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    args.samples_per_gpu = 8
    args.workers_per_gpu = 0
    args.img_range = 2047.0



    sess = PansharpeningSession(args)

    # test_PanCollection(args, sess)
    # test_DLPan(args, sess)
    test_my_data(args, sess)

