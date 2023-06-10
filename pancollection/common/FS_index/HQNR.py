# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

"""
 Description: 
           Hybrid Quality with No Reference (HQNR) index. 

 Interface:
           [HQNR_index,D_lambda,D_S] = HQNR(ps_ms,ms,msexp,pan,S,sensor,ratio)

 Inputs:
           ps_ms:              Pansharpened image;
           ms:                 Original MS image;
           msexp:              MS image resampled to panchromatic scale;
           pan:                Panchromatic image;
           S:                  Block size;
           sensor:             String for type of sensor (e.g. 'WV2','IKONOS');
           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value.
 
 Outputs:
           HQNR_index:          HQNR index;
           D_lambda:            Spectral distortion index;
           D_S:                 Spatial distortion index.

 Notes:
     Results very close to the MATLAB toolbox's ones. In particular, the results of D_S are more accurate than the MATLAB toolbox's ones
     because the Q-index is applied in a sliding window way. Instead, for computational reasons, the MATLAB toolbox uses a distinct block implementation
     of the Q-index.
 
 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.
"""

import numpy as np
from .D_lambda_K import D_lambda_K
# from .D_s import D_s
from .my_D_s import D_s

def HQNR(ps_ms, ms, msexp, pan, S, sensor, ratio):

    D_lambda = D_lambda_K(ps_ms, msexp, ratio, sensor, S) # .copy()
    D_S = D_s(ps_ms, msexp, ms, pan, ratio, S, 1)
    HQNR_index = (1 - D_lambda) * (1 - D_S)

    return HQNR_index, D_lambda, D_S
