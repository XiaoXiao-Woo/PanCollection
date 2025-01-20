<div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://github.com/liangjiandeng/DLPan-Toolbox"><img src="logo/logo-dlpan.png" width="250"></a>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://liangjiandeng.github.io/PanCollection.html"><img src="logo/logo-collection.png" width="250"></a>
</div>


# "PanCollection" for Remote Sensing Pansharpening (Release v1.0.0 [PyPI](https://pypi.org/project/pancollection/) :tada:)

[English](https://github.com/XiaoXiao-Woo/PanCollection/edit/dev/README.md) | [简体中文](https://github.com/XiaoXiao-Woo/PanCollection/edit/dev/README_zh.md)



# Release Notes

The following works is implemented by this repository: 

* **2025.1**: Release **PanCollection v1.0.0**. 🎉
* **2024.12**: *Fully-connected Transformer for Multi-source Image Fusion.*  IEEE T-PAMI 2025. ([Paper coming soon](coming soon)) 📖
* **2024.12**: *Deep Learning in Remote Sensing Image Fusion: Methods, Protocols, Data, and Future Perspectives.* IEEE GRSM 2024. ([Paper](https://ieeexplore.ieee.org/abstract/document/10778974)) 📖
* **2024.10**: *SSDiff: Spatial-spectral Integrated Diffusion Model for Remote Sensing Pansharpening.* NeurIPS 2024. ([Paper](https://openreview.net/pdf?id=QMVydwvrx7), [Code](https://github.com/Z-ypnos/SSDiff_main)) 🚀
* “基于卷积神经网络的遥感图像全色锐化进展综述及相关数据集发布” ([Paper](https://liangjiandeng.github.io/papers/2022/deng-jig2022.pdf), [Homepage](https://liangjiandeng.github.io/PanCollection.html)). 🌐
* **2022.9**: Made available on [PyPI](https://pypi.org/project/pancollection/). 📦
* **2022.9**: Added Colab Demo. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KpWWj1lVUGllZCws01zQfd6CeURuGL2O#scrollTo=k53dsFhAdp6n) ☁️
* **2022.9**: Released the **PanCollection** of the pansharpening training-test dataset for related satellites (such as **WorldView-3**, **QuickBird**, **GaoFen2**, **WorldView-2**). 🛰️
* **2022.5**: Released the Python code based on the unified Pytorch framework, facilitating access for later scholars. 🐍
* **2022.5**: Released a unified pansharpening framework with traditional/deep learning methods (including MATLAB test software package). See [link](https://github.com/liangjiandeng/DLPan-Toolbox/tree/main/02-Test-toolbox-for-traditional-and-DL(Matlab)). ⚙️
* **2021.5**: *Dynamic Cross Feature Fusion for Remote Sensing Pansharpening* accepted by ICCV 2021. ([Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_Dynamic_Cross_Feature_Fusion_for_Remote_Sensing_Pansharpening_ICCV_2021_paper.html), [Code](https://github.com/XiaoXiao-Woo/Dynamic-Cross-Feature-Fusion-for-Remote-Sensing-Pansharpening)) 📚

See the [repo](https://github.com/liangjiandeng/PanCollection) for more detailed descriptions. 

See the [PanCollection Paper](https://liangjiandeng.github.io/papers/2022/deng-jig2022.pdf) for early results.

## Features

| **Features** | **Value** | 
| ------------- | --------- |
| Automatic experimental configuration   | ✅      |   
| Lightning, transformers, accelerate, mmcv, FSDP, DeepSpeed, etc.     | ✅      |   
| Evaluation of Reduced/Full resolution dataset      | ✅      |   
| Multiple models, including FCFormer, SSDiff, CANConv, etc.   | ✅      |   
| Multiple datasets, including WorldView-3, QuickBird, GaoFen-2, WorldView-2, etc.   | ✅      | Training; Testing                 |
| Download and upload huggingface models   | ✅     |            


## Recommendations

We recommend users utilize this code toolbox alongside our other open-source datasets for optimal results:

Python Evaluation: Available in the current repository. For MATLAB Evaluation, refer to the DLPan-Toolbox.
Dataset: Access the PanCollection, which includes the MATLAB test software package in DLPan-Toolbox for fair training and testing.
For Training and Inference, combine UDL with the dataset PanCollection to ensure a fair training and testing environment!

## Datasets (Reduced and Full)

| **Satellite** | **Value** | **Comment**                            |
|--------------------|-----------|----------------------------------------|
| WorldView-3        | 2047      |   Training; Testing; Generalization   |
| QuickBird          | 2047      |    Training; Testing   |
| GaoFen-2           | 1023      |    Training; Testing   |
| WorldView-2        | 2047      |    Training; Testing; Generalization        |



## Easier Quick Start (coming soon)

🤗 To get started with PanCollection benchmark (training, inference, etc.), we recommend reading [Google Colab](https://colab.research.google.com/drive/1KpWWj1lVUGllZCws01zQfd6CeURuGL2O#scrollTo=k53dsFhAdp6n)!



### Set Your Python Environment.

>git clone https://github.com/XiaoXiao-Woo/PanCollection

Then, 

> pip install -e .


or

> pip install -i pancollection https://pypi.org/simple


### Download datasets 
Four satellite datasets (WorldView-3, QuickBird, GaoFen2, WorldView2) are available from the [homepage](https://liangjiandeng.github.io/PanCollection.html). Put it with the following format. 

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
│   │   │   ├── ...
```

### Run the code


coming soon

## Plannings
	
- [ ] [hugging face 🤗](https://huggingface.co/datasets/elsting/PanCollection)
  - [ ] Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the web demo: [Hugging Face Spaces](https://huggingface.co/spaces/elsting/PanCollection)
- [x] Support hydra for parameter setting.
- [ ] Support model zoo with downloading and uploading from [Huggingface 🤗](https://huggingface.co/models?search=pansharpening)
- [ ] Make the [Leaderboard](https://paperswithcode.com/dataset/worldview-3-pancollection) for model results.


## Contribution
We appreciate all contributions to improving PanCollection. Looking forward to your contribution to PanCollection.


## Citation
Please cite this project if you use datasets or the toolbox in your research.

```bibtex
@article{FCFormer,
  title={Fully-connected Transformer for Multi-source  Image Fusion},
  author={Xiao Wu, Zi-Han Cao, Ting-Zhu Huang, Liang-Jian Deng, Jocelyn Chanussot, and Gemine Vivone}
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}
```

```bibtex
@InProceedings{Wu_2021_ICCV,
    author    = {Wu, Xiao and Huang, Ting-Zhu and Deng, Liang-Jian and Zhang, Tian-Jing},
    title     = {Dynamic Cross Feature Fusion for Remote Sensing Pansharpening},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {14687-14696}
}
```


```bibtex
@article{vivone2024deep,
  title={Deep Learning in Remote Sensing Image Fusion: Methods, protocols, data, and future perspectives},
  author={Vivone, Gemine and Deng, Liang-Jian and Deng, Shangqi and Hong, Danfeng and Jiang, Menghui and Li, Chenyu and Li, Wei and Shen, Huanfeng and Wu, Xiao and Xiao, Jin-Liang and others},
  journal={IEEE Geoscience and Remote Sensing Magazine},
  year={2024},
  publisher={IEEE}
}
```

```bibtex
@article{ssdiff,
  title={SSDiff: Spatial-spectral Integrated Diffusion Model for Remote Sensing Pansharpening},
  author={Zhong, Yu and Wu, Xiao and Deng, Liang-Jian and Cao, Zihan},
  journal={arXiv preprint arXiv:2404.11537},
  year={2024}
}
```

```bibtex
@ARTICLE{duancvpr2024,
title={Content-Adaptive Non-Local Convolution for Remote Sensing Pansharpening},
author={Yule Duan, Xiao Wu, Haoyu Deng, Liang-Jian Deng*},
journal={IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR)},
year={2024}
}
```



```bibtex
@ARTICLE{dengjig2022,
	author={邓良剑，冉燃，吴潇，张添敬},
	journal={中国图象图形学报},
	title={遥感图像全色锐化的卷积神经网络方法研究进展},
 	year={2022},
  	volume={},
  	number={9},
  	pages={},
  	doi={10.11834/jig.220540}
   }
```

```bibtex
@ARTICLE{deng2022grsm,
author={L.-J. Deng, G. Vivone, M. E. Paoletti, G. Scarpa, J. He, Y. Zhang, J. Chanussot, and A. Plaza},
booktitle={IEEE Geoscience and Remote Sensing Magazine},
title={Machine Learning in Pansharpening: A Benchmark, from Shallow to Deep Networks},
year={2022},
pages={2-38},
doi={10.1109/MGRS.2020.3019315}
}
```

```bibtex
@misc{PanCollection,
    author = {Xiao Wu, Liang-Jian Deng and Ran Ran},
    title = {"PanCollection" for Remote Sensing Pansharpening},
    url = {https://github.com/XiaoXiao-Woo/PanCollection/},
    year = {2022},
}
```


## Acknowledgement
- [accelerate](https://github.com/huggingface/accelerate): Accelerate is a simple way to train and use PyTorch models with multi-GPU, TPU, and mixed-precision.
- [hydra](https://hydra.cc/): Hydra is a framework for elegantly configuring complex applications.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [UDL](https://github.com/XiaoXiao-Woo/UDL): UDL is a unified framework for vision tasks.with accelerate, lightning, transformers, mmcv1 engines.

## License & Copyright
This project is open sourced under GNU General Public License v3.0.

