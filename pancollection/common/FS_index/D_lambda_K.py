# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu
# @reference:
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )

"""

"""
 Description:
           Spectral distorsion index
 
 Interface:
           Dl = D_lambda_K(fused,ms,ratio,sensor,S)
 
 Inputs:       
       fused:  Pansharpened image;
       ms:     Original MS image;
       ratio:  Scale ratio between MS and PAN. Pre-condition: Integer value;
       sensor: String for type of sensor (e.g. 'WV2','IKONOS');
       S:      Block size.
 
 Output:
       Dl:     Spectral distorsion index.
           
 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.  
"""

from .MTF import MTF
# from .q2n import q2n
from .my_q2n import q2n

def D_lambda_K(fused,ms,ratio,sensor,S):

    if (fused.shape[0] != (ms.shape[0]) or fused.shape[1] != ms.shape[1]) == 1:
        print("The two images must have the same dimensions")
        return -1
    
    # N = fused.shape[0]
    # M = fused.shape[1]
    # if np.remainder(N,S-1) != 0:
    #     print("Number of rows must be multiple of the block size")
    #     return -1    
    # if np.remainder(M,S-1) != 0:
    #     print("Number of columns must be multiple of the block size")
    #     return -1

    fused_degraded = MTF(fused,sensor,ratio)
                
    # fused_degraded = fused_degraded[int(ratio/2):-1:int(ratio),int(ratio/2):-1:int(ratio),:]
        
    Q2n_index, _ = q2n(ms, fused_degraded, S, S)
        
    Dl = 1 - Q2n_index
    
    return Dl
