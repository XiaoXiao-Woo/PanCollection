import glob
import torch
from torch.utils.data import DataLoader
from udl_vis.Basis.distributed import DistributedSampler, RandomSampler
from pancollection.common.test_panloader import PanCollection_dataset
from pancollection.common.train_panloader import PanCollection_train_dataset
from pancollection.common.valid_panloader import PanCollection_valid_dataset
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

    def get_train_dataloader(self, dataset_name, distributed, state_dataloader):

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

    def get_test_dataloader(self, dataset_name, distributed):

        dataset_type = self.args.dataset_type
        if dataset_type == "oldPan":
            dataset = oldPan_dataset(dataset_name, self.args.dataset)
        elif dataset_type == "DLPan":
            dataset = DLPan_dataset(dataset_name, self.args.dataset)
        elif dataset_type == "PanCollection":
            dataset = PanCollection_dataset(
                dataset_name, self.args.dataset, self.args.img_scale[dataset_name]
            )
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
