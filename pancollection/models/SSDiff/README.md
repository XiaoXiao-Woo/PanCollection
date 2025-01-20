# SSDiff: Spatial-spectral Integrated Diffusion Model for Remote Sensing Pansharpening [NeurIPS 2024]

<div style="text-align: center;">
  <a href="https://www.arxiv.org/">
    <img src="https://img.shields.io/badge/arXiv-red.svg?style=flat" alt="ArXiv">
  </a>
    <a href="https://arxiv.org/abs/2404.11537">Arxiv</a>
    <img class="nips-logo" src="https://neurips.cc/static/core/img/NeurIPS-logo.svg" alt="NeurIPS logo" height="25" width="58">
    <a href="https://openreview.net/pdf?id=QMVydwvrx7">NeurIPS 2024
</div>
<p style="text-align: center; font-family: 'Times New Roman';">
  </a>
    Abstract:
    Pansharpening is a significant image fusion technique that merges the spatial content and spectral characteristics of remote sensing images to generate high-resolution multispectral images. Recently, denoising diffusion probabilistic models have been gradually applied to visual tasks, enhancing controllable image generation through low-rank adaptation (LoRA). 
    In this paper, we introduce a spatial-spectral integrated diffusion model for the remote sensing pansharpening task, called SSDiff, which considers the pansharpening process as the fusion process of spatial and spectral components from the perspective of subspace decomposition. 
    Specifically, SSDiff utilizes spatial and spectral branches to learn spatial details and spectral features separately, then employs a designed alternating projection fusion module (APFM) to accomplish the fusion. Furthermore, we propose a frequency modulation inter-branch module (FMIM) to modulate the frequency distribution between branches. 
    The two components of SSDiff can perform favorably against the APFM when utilizing a LoRA-like branch-wise alternative fine-tuning method. It refines SSDiff to capture component-discriminating features more sufficiently. 
    Finally, extensive experiments on four commonly used datasets, i.e., WorldView-3, WorldView-2, GaoFen-2, and QuickBird, demonstrate the superiority of SSDiff both visually and quantitatively.
</a>
</p>

News:
- 2024/11/7：**Code RELEASED!**:fire: 

- 2024/10/18: **Code will be released soon!**:fire: 

## Quick Overview

The code in this repo supports Pansharpening.

<table><tr>
<td><img src="https://github.com/Z-ypnos/blog_img_bed/blob/main/ssdiff/ssdiff_model.png" border=0></td>
</tr></table>

# Instructions

## Dataset

In this office repo, you can find the Pansharpening dataset of [WV3, GF2, and QB](https://github.com/liangjiandeng/PanCollection).

Other instructions will come soon!


## Citation

If you find our paper useful, please consider citing the following:

```tex
@article{zhong2024ssdiff,
  title={SSDiff: Spatial-spectral Integrated Diffusion Model for Remote Sensing Pansharpening},
  author={Zhong, Yu and Wu, Xiao and Deng, Liang-Jian and Cao, Zihan},
  journal={arXiv preprint arXiv:2404.11537},
  year={2024}
}
```
