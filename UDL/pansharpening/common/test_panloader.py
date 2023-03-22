# GPL License
# Copyright (C) 2022 , UESTC
# All Rights Reserved 
#
# @Time    : 2022/11/30 21:06
# @Author  : Xiao Wu
# @reference: 
#
import glob


def oldPan_dataloader(dataset_name, args):
    if 'new_data' in dataset_name:
        from UDL.pansharpening.evaluation.ps_evaluate import SingleDataset
        dataset = SingleDataset(['/'.join([args.data_dir, "oldPan/test_data", f"{dataset_name}.mat"])], dataset_name,
                                img_scale=args.img_range)
    elif 'gf81' in dataset_name or 'qb48' in dataset_name:
        # from UDL.pansharpening.evaluation.ps_evaluate import SingleDataset
        # dataset = SingleDataset(['/'.join([args.data_dir, "oldPan/test_data/gf2/", f"{dataset_name}"])], dataset_name,
        #                         img_scale=args.img_range)
        from UDL.pansharpening.evaluation.ps_evaluate import MultiExmTest_h5
        dataset = MultiExmTest_h5(['/'.join([args.data_dir, "oldPan/test_data/gf2/", f"{dataset_name}"])],
                                  dataset_name, suffix='.mat', img_scale=args.img_range)

    # elif 'mulExm1258' in dataset_name:
    #     from UDL.pansharpening.evaluation.ps_evaluate import MultiExmTest_h5
    #     dataset = MultiExmTest_h5('/'.join([args.data_dir, f"oldPan/test_data/{dataset_name}"]),
    #                               dataset_name, suffix='.mat', img_scale=args.img_range)

    elif 'singleMat' in dataset_name:
        from UDL.pansharpening.evaluation.ps_evaluate import SingleDataset
        dataset = SingleDataset(glob.glob('/'.join([args.data_dir, "test_data/singleMat", "*.mat"])), dataset_name,
                                img_scale=args.img_range)
    elif 'TestData_QB' in dataset_name:
        from UDL.pansharpening.evaluation.ps_evaluate import SingleDataset
        dataset = SingleDataset(glob.glob('/'.join([args.data_dir, "oldPan/test_data", "TestData_QB.mat"])), dataset_name,
                                img_scale=args.img_range)
    elif 'mulExm' in dataset_name:
        satellite = dataset_name.split('_')[1]
        suffix = dataset_name.split('.')[-1]
        from UDL.pansharpening.evaluation.ps_evaluate import MultiExmTest_h5
        dataset = MultiExmTest_h5('/'.join([args.data_dir, f"oldPan/test_data/{satellite}/{dataset_name}"]),
                                  dataset_name, suffix=f'.{suffix}', img_scale=args.img_range)
    else:
        dataset = None

    return dataset


def DLPan_dataloader(dataset_name, args):
    if 'valid' in dataset_name:
        if "hp" in dataset_name:
            from UDL.pansharpening.common.dataset_hp import Dataset_Pro
            dataset = Dataset_Pro(
                '/'.join([args.data_dir, 'validation_data', f'{dataset_name}']), img_scale=args.img_range)

        else:
            from UDL.pansharpening.common.dataset import Dataset_Pro
            dataset = Dataset_Pro('/'.join([args.data_dir, 'validation_data', f'{dataset_name}']),
                                  img_scale=args.img_range)

    elif 'TestData' in dataset_name:
        if 'hp' in dataset_name:
            satellite = dataset_name.split('_')[-2]
        else:
            satellite = dataset_name.split('_')[-1]
            satellite = satellite.replace('.h5', '')

        from UDL.pansharpening.evaluation.ps_evaluate import MultiExmTest_h5
        dataset = MultiExmTest_h5(
            '/'.join([args.data_dir, 'test_data', satellite.lower(), f"{dataset_name.replace('_hp', '')}"]),
            dataset_name, img_scale=args.img_range)

    elif 'RR' in dataset_name or 'FR' in dataset_name:
        splits = dataset_name.split('_')
        if 'hp' in dataset_name:
            satellite = splits[-3]
        else:
            satellite = splits[-2]

        from UDL.pansharpening.evaluation.ps_evaluate import SingleDataset

        dataset = SingleDataset(['/'.join([args.data_dir, 'test_data', satellite.lower(),
                                           dataset_name.replace('_hp', '')])], dataset_name,
                                img_scale=args.img_range)
    else:
        dataset = None

    return dataset


def PanCollection_dataloader(dataset_name, args):
    if 'Test(HxWxC)' in dataset_name:
        # Test(HxWxC)_gf2_data_fr/rr...
        from UDL.pansharpening.evaluation.ps_evaluate import SingleDatasetV2
        satellite = dataset_name.split('_')[1]
        type = 'FR-Data' if 'fr' in dataset_name else 'RR-Data'
        dataset = SingleDatasetV2(
            glob.glob('/'.join([args.data_dir, f"/test_data/{satellite.upper()}/{type}/*.mat"])), dataset_name,
            img_scale=args.img_range)

    elif 'multiExm' in dataset_name:
        suffix = dataset_name.split('.')[-1]
        from UDL.pansharpening.evaluation.ps_evaluate import MultiExmTest_h5
        dataset = MultiExmTest_h5('/'.join([args.data_dir, f"PanCollection/test_data/{dataset_name}"]),
                                  dataset_name, suffix=f'.{suffix}', img_scale=args.img_range)
    else:
        dataset = None

    return dataset
