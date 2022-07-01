# PanCollection
"PanCollection" for Remote Sensing Pansharpening

[English](https://github.com/XiaoXiao-Woo/PanCollection/edit/dev/README.md) | [简体中文](https://github.com/XiaoXiao-Woo/PanCollection/edit/dev/README_zh.md)

This repository is the official PyTorch implementation of “基于卷积神经网络的遥感图像全色锐化进展综述及相关数据集发布” ([paper](), [homepage](https://liangjiandeng.github.io/PanCollection.html).

* Release the PanCollection of the pan-sharpening training-test dataset of related satellites (such as WorldView-3, QuickBird, GaoFen2, WorldView-2 satellites); 
* Release the Python code unified writing framework based on the Pytorch deep learning library, which is convenient for later scholars;
* Release a unified Pansharpening traditional-deep learning method ( including MATLAB test software package), which is convenient for later scholars to conduct fair tests;

## Requirements
* Python3.7+, Pytorch>=1.6.0
* NVIDIA GPU + CUDA
* Run `python setup.py develop`

Note: Our project is based on MMCV, but you needn't install it currently.

## Quick Start
**Step0.** set your Python environment.

>git clone https://github.com/XiaoXiao-Woo/PanCollection
> python setup.py develop

**Step1.**
Download Datasets (WorldView-3, QuickBird, GaoFen2, WorldView2) from the [homepage](PanCollection for Survey Paper (liangjiandeng.github.io)](https://liangjiandeng.github.io/PanCollection.html)). Put it with the following format.

```
|-$ROOT/Datasets
├── pansharpening
│   ├── training_data
│   │   ├── train_wv3.h5
│   │   ├── ...
│   ├── validation_data
│   │   │   ├── valid_wv3.h5
│   │   │   ├── ...
│   ├── test_data
│   │   ├── WV3
│   │   │   ├── test_wv3_multiExm.h5
│   │   │   ├── test_wv3_multiExm.h5
│   │   │   ├── ...
```

**Step2.** Open PanCollection/UDL/pansharpening,  run the following code:

> python run_pansharpening.py

**step3.** How to train/test the code.

* A training example：

	run_pansharpening.py
  
	where arch='BDPN', and option_bdpn.py has: 
  
	cfg.eval = False, 
  
	cfg.workflow = [('train', 50), ('val', 1)], cfg.dataset = {'train': 'wv3', 'val': 'wv3_multiExm.h5'}
	
* A test example:

	run_test_pansharpening.py
  
	cfg.eval = True or cfg.workflow = [('val', 1)]

**Step4**. How to customize the code

One model is divided into three parts.

1. Option_*modelName*.py records hyperparameter configures in folder of PanCollection/UDL/pansharpening/configs.

2. Set model, loss, optimizer, scheduler. see in folder of PanCollection/UDL/pansharpening/models/*modelName*_main.py.

3. write a new model in folder of PanCollection/UDL/pansharpening/models/*modelName*/model_*modelName*.py.

Note that when you add a new model into PanCollection, you need to update  PanCollection/UDL/pansharpening/models/`__init__.py` and add option_*modelName*

**Others**
* if you want to add customized datasets, you need to update:

>PanCollection/UDL/AutoDL/`__init__.py`.
>PanCollection/UDL/pansharpening/common/psdata.py.

* if you want to add customized tasks, you need to update:

> PanCollection/UDL/*taskName*/models to put model_*newModelName* and *newModelName*_main.
> Create a new folder of PanCollection/UDL/*taskName*/configs to put option__*newModelName*.

>PanCollection/UDL/AutoDL/`__init__.py`.
>add a class in PanCollection/UDL/Basis/python_sub_class.py, like this:
```class PanSharpeningModel(ModelDispatcher, name='pansharpening'):```

* if you want to add customized training settings, such as saving model, recording logs, and so on. you need to update:

> PanCollection/UDL/mmcv/mmcv/runner/hooks

Note that: Don't put model/dataset/task-related files into the folder of AutoDL.

* if you want to know runner how to train/test in PanCollection/UDL/AutoDL/trainer.py, please see PanCollection/UDL/mmcv/mmcv/runner/epoch_based_runner.py


## Contribution
We appreciate all contributions to improving PanCollection. Looking forward to your contribution to PanCollection.


## Citation
Please cite this project if you use datasets or the toolbox in your research.
> 


## Acknowledgement
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.

## License & Copyright
This project is open sourced under GNU General Public License v3.0.

