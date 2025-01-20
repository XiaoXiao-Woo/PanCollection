# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""
"""
 Description: 
           interp23 interpolates the image I_Interpolated using a polynomial with 23 coefficients interpolator. 
 
 Interface:
           image = interp23(image,ratio)

 Inputs:
           image:          Image to interpolate;
           ratio:          Scale ratio between MS and PAN. Pre-condition: Resize factors power of 2.

 Outputs:
           image:          Interpolated image.
 
 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.
"""

import numpy as np
from scipy import ndimage

def interp23(image, ratio):
    if (2**round(np.log2(ratio)) != ratio):
        print("Error: only resize factors of power 2")
        return -1

    r = image.shape[0]
    c = image.shape[1]
    
    if (np.size(image.shape) == 3):      
        b = image.shape[2]
    else:
        b = 1
    
    CDF23 = 2*np.array([0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0, -0.000060081482])
    d = CDF23[::-1] 
    CDF23 = np.insert(CDF23, 0, d[:-1])
    BaseCoeff = CDF23
    
    first = 1
    for z in range(1,np.int(np.log2(ratio))+1):
        if (b == 1):
            I1LRU = np.zeros(((2**z)*r, (2**z)*c))
        else:
            I1LRU = np.zeros(((2**z)*r, (2**z)*c, b))
            
        if first:
            if (b == 1):
                I1LRU[1:I1LRU.shape[0]:2,1:I1LRU.shape[1]:2]=image
            else:
                I1LRU[1:I1LRU.shape[0]:2,1:I1LRU.shape[1]:2,:]=image
            first = 0
        else:
            if (b == 1):
                I1LRU[0:I1LRU.shape[0]:2,0:I1LRU.shape[1]:2]=image
            else:
                I1LRU[0:I1LRU.shape[0]:2,0:I1LRU.shape[1]:2,:]=image
        
        for ii in range(b):
            if (b == 1):
                t = I1LRU
            else:
                t = I1LRU[:,:,ii]
                
            for j in range(0,t.shape[0]):
                t[j,:]=ndimage.correlate(t[j,:],BaseCoeff,mode='wrap')
            for k in range(0,t.shape[1]):
                t[:,k]=ndimage.correlate(t[:,k],BaseCoeff,mode='wrap')
            if (b == 1):
                I1LRU = t
            else:
                I1LRU[:,:,ii] = t
            
        image = I1LRU
        
    return image