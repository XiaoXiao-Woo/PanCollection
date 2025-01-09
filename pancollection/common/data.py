import torch
from torch.utils.data import Dataset, DataLoader
from udl_vis.Basis.distributed import DistributedSampler, RandomSampler


class DummyPansharpeningDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.ms = torch.randn(num_samples, 8, 64, 64)
        self.gt = torch.randn(num_samples, 8, 256, 256)
        self.lms = torch.randn(num_samples, 8, 256, 256)
        self.pan = torch.randn(num_samples, 1, 256, 256)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "ms": self.ms[idx],
            "gt": self.gt[idx],
            "lms": self.lms[idx],
            "pan": self.pan[idx],
        }


class DummySession:
    def __init__(self, args):
        self.dataloaders = {}
        self.samples_per_gpu = args.samples_per_gpu
        self.workers_per_gpu = args.workers_per_gpu
        self.writers = {}
        self.args = args

    def get_dataloader(self, distributed):

        sampler = None
        generator = None
        dataset = DummyPansharpeningDataset()

        if distributed:
            sampler = DistributedSampler(dataset)
        else:
            sampler = RandomSampler(dataset)

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

        return dataloaders, sampler, generator

    def get_eval_dataloader(self, distributed):

        sampler = None
        dataset = DummyPansharpeningDataset()
        dataloaders = DataLoader(
            dataset,
            batch_size=self.args.test_samples_per_gpu,
            shuffle=False,
            num_workers=1,
            drop_last=False,
            sampler=sampler,
        )

        return dataloaders, sampler
