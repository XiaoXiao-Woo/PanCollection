# U2Net: A General Framework with Spatial-Spectral-Integrated Double U-Net for Image Fusion
- Code for the paper: "U2Net: A General Framework with Spatial-Spectral-Integrated Double U-Net for Image Fusion", ACM MM 2023.

- State-of-the-art (SOTA) performance on the [PanCollection](https://github.com/liangjiandeng/PanCollection) of remote sensing pansharpening.

# Method
## Overall Structure

![u2net](show_image/u2net.png#pic_center)

We employ a spatial U-Net and a spectral U-Net to extract spatial details and spectral characteristics, respectively. Besides, feature maps from different sources are integrated through the well-designed S2Blocks.

## S2Block

![s2block](show_image/s2block.png#pic_center)

The S2Block merges feature maps from different sources in a sufficient manner.

# Experimental Results
## Pansharpening
- Quantitative evalutaion results on WV3 datasets of PanCollection.

![wv3](show_image/wv3.png#pic_center)
- Qualitative evalutaion results on the WV3 dataset of PanCollection.

![wv3](show_image/wv3_visual.png#pic_center)

## HISR
- Qualitative evalutaion results on the CAVE dataset.

![cave](show_image/cave_visual.png#pic_center)

# Get Started
## Dataset
- Datasets for pansharpening: [PanCollection](https://github.com/liangjiandeng/PanCollection). The downloaded data can be placed everywhere because we do not use relative path. Besides, we recommend the h5py format, as if using the mat format, the data loading section needs to be rewritten.

- Dataset for HISR: the CAVE dataset. You can find this dataset on the Internet and simulate it using the Wald's protocal.

## Installation and Requirements
```shell
git clone https://github.com/PSRben/U2Net.git
cd U2Net
pip install -r requirements.txt
```

## Usage
- This code is for pansharpening. If you want to apply our method to HISR and other image fusion tasks, please modify the input channels in [this file](model/u2net.py).

- The model weight trained on the WV3 dataset for 200000 iterations can be found [here](weights/).

```shell
# train
python train.py --train_data_path ./path_to_data/train_WV3.h5 --val_data_path ./path_to_data/valid_WV3.h5
# test
python test.py --file_path ./path_to_data/name.h5 --save_dir ./path_to_dir --weight ./weights/epochs.pth
```

# Citation
```shell
@inproceedings{10.1145/3581783.3612084,
author = {Peng, Siran and Guo, Chenhao and Wu, Xiao and Deng, Liang-Jian},
title = {U2Net: A General Framework with Spatial-Spectral-Integrated Double U-Net for Image Fusion},
year = {2023},
isbn = {9798400701085},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3581783.3612084},
doi = {10.1145/3581783.3612084},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {3219–3227},
numpages = {9},
location = {Ottawa ON, Canada},
series = {MM '23}
}

```

# Contact
We are glad to hear from you. If you have any questions, please feel free to contact siran_peng@163.com.
