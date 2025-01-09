## [SSconv: Explicit Spectral-to-Spatial Convolution for Pansharpening](https://github.com/liangjiandeng/liangjiandeng.github.io/tree/master/papers/2021/mucnn_mm2021.pdf)

**Homepage:** https://liangjiandeng.github.io/

- Code for paper: "SSconv: Explicit Spectral-to-Spatial Convolution for Pansharpening, ACMMM, 2021"
- State-of-the-art pansharpening performance


# Method

![flowchart](images_for_show/02-flowchart-MUCNN.png)

# Pytorch training and testing code 2021/07/29

- The proposed network is trained on Pytorch 1.8.0. Because we use GPU CUDA for training by default, the modules about CUDA in the code may need to be adjusted according to your computer.

- The code for training is in `train.py`,while the code for test on one image (.mat) is in `main_test_single.py` and the code for test on multi images (.mat) is in `main_test_multi.py`.

- For training, you need to set the file_path in the main function, adopt to your train set, validate set, and test set as well. Our code train the .h5 file, you may change it through changing the code in main function.

- As for testing, you need to set the path in both main and test function to open and load the file.

- Because we use GPU CUDA for training by default, the modules about CUDA in the code may need to be adjusted according to your computer.

- Datasets: you may find source training and testing code from the folder. However, due to the copyright of dataset, we can not upload the datasets, you may download the data and simulate them according to the paper.

# Pansharpening results

### Quantity results

**Test on the simulated images from WV3**

![1627557326610](images_for_show/1627557326610.png)

**Test on the full resolution images from WV3**

![1627557378337](images_for_show/1627557378337.png)

**Test on the data from GaoFen-2**

![1627557443029](images_for_show/1627557443029.png)

### Visual results

![1627557509836](images_for_show/1627557509836.png)

![1627557529777](images_for_show/1627557529777.png)

![1627557542133](images_for_show/1627557542133.png)

# Citation

```bibtex
@ARTICLE{mucnn,
author={Yudong Wang, Liang-Jian Deng, Tian-Jing Zhang, Xiao Wu},
booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
title={SSconv: Explicit Spectral-to-Spatial Convolution for Pansharpening},
year={2021},
pages={DOI: 10.1145/3474085.3475600.},
}
```

