# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu
# @reference:
import glob


def oldPan_train_dataset(dataset_name, args):

    dataset = None
    if any(list(map(lambda x: x in dataset_name, ["wv2", "wv3", "qb", "gf2"]))):
        if "hp" in dataset_name:
            # high-pass filter
            from pancollection.common.dataset_hp import Dataset_Pro
        else:
            from pancollection.common.dataset import Dataset_Pro

        if "wv3" in dataset_name:
            dataset = Dataset_Pro(
                "/".join([args.data_dir, "oldPan/training_data/train_wv3_8806.h5"]),
                img_scale=args.img_range,
            )
        elif "qb" in dataset_name:
            dataset = Dataset_Pro(
                "/".join([args.data_dir, "oldPan/training_data/train_qb_20685.h5"]),
                img_scale=args.img_range,
            )
        elif "gf2" in dataset_name:
            dataset = Dataset_Pro(
                "/".join([args.data_dir, "oldPan/training_data/train_gf2_21607.h5"]),
                img_scale=args.img_range,
            )
    else:
        print(f"{dataset_name} is not supported.")
        raise NotImplementedError

    return dataset


def DLPan_train_dataset(dataset_name, args):

    dataset = None
    if any(list(map(lambda x: x in dataset_name, ["wv2", "wv3", "wv4", "qb"]))):
        if "hp" in dataset_name:
            # high-pass filter
            from pancollection.common.dataset_hp import Dataset_Pro
        else:
            from pancollection.common.dataset import Dataset_Pro

        if "wv3" in dataset_name:
            dataset = Dataset_Pro(
                "/".join([args.data_dir, "DLPan/training_data/train_wv3_10000.h5"]),
                img_scale=args.img_range,
            )
        elif "wv2" in dataset_name:
            dataset = Dataset_Pro(
                "/".join([args.data_dir, "DLPan/training_data/train_wv2_10000.h5"]),
                img_scale=args.img_range,
            )
        elif "qb" in dataset_name:
            dataset = Dataset_Pro(
                "/".join([args.data_dir, "DLPan/training_data/train_qb_10000.h5"]),
                img_scale=args.img_range,
            )
        elif "wv4" in dataset_name:
            dataset = Dataset_Pro(
                "/".join([args.data_dir, "DLPan/training_data/train_wv4_10000.h5"]),
                img_scale=args.img_range,
            )
    else:
        print(f"{dataset_name} is not supported.")
        raise NotImplementedError

    return dataset


def PanCollection_train_dataset(dataset_name, args):

    dataset = None
    if any(list(map(lambda x: x in dataset_name, ["wv2", "wv3", "qb", "gf2"]))):
        if "hp" in dataset_name:
            # high-pass filter
            from pancollection.common.dataset_hp import Dataset_Pro
        else:
            from pancollection.common.dataset import Dataset_Pro

        if "wv3" in dataset_name:
            dataset = Dataset_Pro(
                args.dataset.wv3_train_path,
                img_scale=args.img_range,
            )
        elif "wv2" in dataset_name:
            dataset = Dataset_Pro(
                args.dataset.wv2_train_path,
                img_scale=args.img_range,
            )
        elif "qb" in dataset_name:
            dataset = Dataset_Pro(
                args.dataset.qb_train_path,
                img_scale=args.img_range,
            )
        elif "gf2" in dataset_name:
            dataset = Dataset_Pro(
                args.dataset.gf2_train_path,
                img_scale=args.img_range,
            )
    else:
        print(f"{dataset_name} is not supported.")
        raise NotImplementedError

    return dataset
