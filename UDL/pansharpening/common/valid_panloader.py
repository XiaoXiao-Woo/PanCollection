# GPL License
# Copyright (C) 2022 , UESTC
# All Rights Reserved 
#
# @Time    : 2022/11/30 21:07
# @Author  : Xiao Wu
# @reference:
#
# GPL License
# Copyright (C) 2022 , UESTC
# All Rights Reserved
#
# @Time    : 2022/11/30 21:06
# @Author  : Xiao Wu
# @reference:
#
import glob


def oldPan_trainloader(dataset_name, args):

    dataset = None
    if any(list(map(lambda x: x in dataset_name, ['wv2', 'wv3', 'qb', 'gf2']))):
        if "hp" in dataset_name:
            # high-pass filter
            from UDL.pansharpening.common.dataset_hp import Dataset_Pro
        else:
            from UDL.pansharpening.common.dataset import Dataset_Pro

        if "wv3" in dataset_name:
            dataset = Dataset_Pro('/'.join([args.data_dir, 'oldPan/validation_data/valid_wv3_8806.h5']),
                                  img_scale=args.img_range)
        elif "qb" in dataset_name:
            dataset = Dataset_Pro('/'.join([args.data_dir, 'oldPan/validation_data/valid_qb_20685.h5']),
                                  img_scale=args.img_range)
        elif "gf2" in dataset_name:
            dataset = Dataset_Pro('/'.join([args.data_dir, 'oldPan/validation_data/valid_gf2_21607.h5']),
                                  img_scale=args.img_range)
    else:
        print(f"{dataset_name} is not supported.")
        raise NotImplementedError

    return dataset


def DLPan_trainloader(dataset_name, args):

    dataset = None
    if any(list(map(lambda x: x in dataset_name, ['wv2', 'wv3', 'wv4', 'qb']))):
        if "hp" in dataset_name:
            # high-pass filter
            from UDL.pansharpening.common.dataset_hp import Dataset_Pro
        else:
            from UDL.pansharpening.common.dataset import Dataset_Pro

        if "wv3" in dataset_name:
            dataset = Dataset_Pro('/'.join([args.data_dir, 'DLPan/validation_data/valid_wv3_10000.h5']),
                                  img_scale=args.img_range)
        elif "wv2" in dataset_name:
            dataset = Dataset_Pro('/'.join([args.data_dir, 'DLPan/validation_data/valid_wv2_10000.h5']),
                                  img_scale=args.img_range)
        elif "qb" in dataset_name:
            dataset = Dataset_Pro('/'.join([args.data_dir, 'DLPan/validation_data/valid_qb_10000.h5']),
                                  img_scale=args.img_range)
        elif "wv4" in dataset_name:
            dataset = Dataset_Pro('/'.join([args.data_dir, 'DLPan/validation_data/valid_wv4_10000.h5']),
                                  img_scale=args.img_range)
    else:
        print(f"{dataset_name} is not supported.")
        raise NotImplementedError

    return dataset


def PanCollection_trainloader(dataset_name, args):

    dataset = None
    if any(list(map(lambda x: x in dataset_name, ['wv2', 'wv3', 'qb', 'gf2']))):
        if "hp" in dataset_name:
            # high-pass filter
            from UDL.pansharpening.common.dataset_hp import Dataset_Pro
        else:
            from UDL.pansharpening.common.dataset import Dataset_Pro

        if "wv3" in dataset_name:
            dataset = Dataset_Pro('/'.join([args.data_dir, 'PanCollection/validation_data/valid_wv3.h5']),
                                  img_scale=args.img_range)
        elif "wv2" in dataset_name:
            dataset = Dataset_Pro('/'.join([args.data_dir, 'PanCollection/validation_data/valid_wv2.h5']),
                                  img_scale=args.img_range)
        elif "qb" in dataset_name:
            dataset = Dataset_Pro('/'.join([args.data_dir, 'PanCollection/validation_data/valid_qb.h5']),
                                  img_scale=args.img_range)
        elif "gf2" in dataset_name:
            dataset = Dataset_Pro('/'.join([args.data_dir, 'PanCollection/validation_data/valid_gf2.h5']),
                                  img_scale=args.img_range)
    else:
        print(f"{dataset_name} is not supported.")
        raise NotImplementedError

    return dataset
