# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu
# @reference:
#
def PanCollection_dataset(dataset_name, dataset, img_scale):
    from pancollection.common.dataset import MultiExmTest_h5

    dataset = MultiExmTest_h5(
        getattr(dataset, f"{dataset_name}_test_path"), dataset_name, img_scale=img_scale
    )

    return dataset


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # file_path = "/Data/Datasets/pansharpening_2/PanCollection/test_data/test_wv3_multiExm1.h5"
    file_path = "/Data/Datasets/pansharpening_2/PanCollection/test_data/WV3/RR-Data/Test(HxWxC)_wv3_data10.mat"

    dataset = PanCollection_dataset({"wv3_test_path": file_path})
    dataloaders = DataLoader(
        dataset,
        batch_size=20,
        persistent_workers=True,
        pin_memory=True,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

    batch = next(iter(dataloaders))
    print(batch.keys())
