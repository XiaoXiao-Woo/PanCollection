import glob
import torch
from torch.utils.data import DataLoader
from udl_vis.Basis.distributed import DistributedSampler, RandomSampler
from pancollection.common.test_panloader import (
    oldPan_dataset,
    PanCollection_dataset,
    DLPan_dataset,
)
from pancollection.common.train_panloader import (
    oldPan_train_dataset,
    PanCollection_train_dataset,
    DLPan_train_dataset,
)
from pancollection.common.valid_panloader import (
    oldPan_valid_dataset,
    PanCollection_valid_dataset,
    DLPan_valid_dataset,
)
from pancollection.common.data import DummyPansharpeningDataset


class PansharpeningSession:
    def __init__(self, args):
        self.dataloaders = {}
        self.samples_per_gpu = args.samples_per_gpu
        self.workers_per_gpu = args.workers_per_gpu
        # self.patch_size = args.patch_size
        self.writers = {}
        self.args = args

        # self.mapping = {'wv3': 'wv3_multiExm1.h5', 'wv2': 'wv2_multiExm1.h5', 'qb': 'qb_multiExm1.h5', 'gf2': 'gf2_multiExm1.h5'}

    def get_dataloader(self, dataset_name, distributed, state_dataloader):

        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        if state_dataloader is not None:
            generator.set_state(state_dataloader.cpu())

        dataset_type = self.args.dataset_type
        if dataset_type == "oldPan":
            dataset = oldPan_train_dataset(dataset_name, self.args)
        elif dataset_type == "DLPan":
            dataset = DLPan_train_dataset(dataset_name, self.args)
        elif dataset_type == "PanCollection":
            dataset = PanCollection_train_dataset(dataset_name, self.args)
        elif dataset_type == "Dummy":
            dataset = DummyPansharpeningDataset()
        else:
            raise NotImplementedError(
                f"{dataset_name} or {dataset_type} is not supported."
            )

        sampler = None
        # # print("distributed: ", distributed)
        # if distributed:
        #     sampler = DistributedSampler(dataset, generator=generator)
        # else:
        #     sampler = RandomSampler(dataset, generator=generator)
        # if distributed:
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        dataloaders = DataLoader(
            dataset,
            batch_size=self.samples_per_gpu,
            persistent_workers=(True if self.workers_per_gpu > 0 else False),
            pin_memory=True,
            shuffle=(sampler is None),
            num_workers=self.workers_per_gpu,
            drop_last=True,
            prefetch_factor=self.workers_per_gpu if self.workers_per_gpu > 0 else None,
            sampler=sampler,
        )

        return dataloaders, sampler, generator

    def get_valid_dataloader(self, dataset_name, distributed):

        dataset = None
        dataset_type = self.args.dataset_type
        if dataset_type == "oldPan":
            dataset = oldPan_valid_dataset(dataset_name, self.args)
        elif dataset_type == "DLPan":
            dataset = DLPan_valid_dataset(dataset_name, self.args)
        elif dataset_type == "PanCollection":
            dataset = PanCollection_valid_dataset(dataset_name, self.args)
        elif dataset_type == "Dummy":
            dataset = DummyPansharpeningDataset(num_samples=10)
        else:
            raise NotImplementedError(
                f"{dataset_name} or {dataset_type} is not supported."
            )

        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=False)
        else:
            sampler = RandomSampler(dataset)
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        # if not dataset_name in self.dataloaders:
        dataloaders = DataLoader(
            dataset,
            batch_size=self.samples_per_gpu,
            persistent_workers=(True if self.workers_per_gpu > 0 else False),
            pin_memory=True,
            shuffle=(sampler is None),
            num_workers=self.workers_per_gpu,
            drop_last=True,
            sampler=sampler,
        )

        return dataloaders, sampler

    def get_eval_dataloader(self, dataset_name, distributed):

        dataset_type = self.args.dataset_type
        if dataset_type == "oldPan":
            dataset = oldPan_dataset(dataset_name, self.args.dataset)
        elif dataset_type == "DLPan":
            dataset = DLPan_dataset(dataset_name, self.args.dataset)
        elif dataset_type == "PanCollection":
            dataset = PanCollection_dataset(self.args.dataset)
        elif dataset_type == "Dummy":
            dataset = DummyPansharpeningDataset(num_samples=10)
        else:
            raise NotImplementedError(
                f"{dataset_name} or {dataset_type} is not supported in choices ['oldPan', 'DLPan', 'PanCollection', 'Dummy']."
            )

        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=False)
        else:
            sampler = RandomSampler(dataset)
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        # if not dataset_name in self.dataloaders:
        dataloaders = DataLoader(
            dataset,
            batch_size=self.args.test_samples_per_gpu,
            shuffle=False,
            num_workers=1,
            drop_last=False,
            sampler=sampler,
        )
        return dataloaders, sampler

    """
     crate_gdfn_l2 - 2024-03-19-21-39-03,0:0:00:06.042322 - Iter [0] Epoch(test) [0]/[1] [1/4]	SAM: 57.29578, ERGAS: 43.99768, PSNR: 13.48509
     crate_gdfn_l2 - 2024-03-19-21-39-03,0:0:00:06.162003 - Iter [0] Epoch(test) [0]/[1] [2/4]	SAM: 57.29578, ERGAS: 53.58638, PSNR: 14.03813
     crate_gdfn_l2 - 2024-03-19-21-39-03,0:0:00:06.275698 - Iter [0] Epoch(test) [0]/[1] [3/4]	SAM: 57.29578, ERGAS: 61.50018, PSNR: 13.51857
     crate_gdfn_l2 - 2024-03-19-21-39-03,0:0:00:06.386402 - Iter [0] Epoch(test) [0]/[1] [4/4]	SAM: 57.29578, ERGAS: 46.04091, PSNR: 14.15239
     crate_gdfn_l2 - 2024-03-19-21-39-03,0:0:00:06.651716 - test time: 6.260583162307739
     crate_gdfn_l2 - 2024-03-19-21-39-03,0:0:00:06.652689 - Iter [0] Epoch(test) [0]/[1] [4/4]	SAM: 57.29578, ERGAS: 51.28128, PSNR: 13.79855
    """


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

    args.dataset = "test_gf2_mulExm_84.h5"
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
    args.dataset = "train_gf2.h5"
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

    args.dataset = "TestData_wv3.h5"
    # args.dataset = 'NY1_WV3_FR.mat'
    loader, _ = sess.get_eval_dataloader(args.dataset, False)

    print(len(loader))


if __name__ == "__main__":
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
