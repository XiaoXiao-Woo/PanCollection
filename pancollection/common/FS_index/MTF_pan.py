# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

"""
 Description: 
           MTF filters the image I_MS using a Gaussin filter matched with the Modulation Transfer Function (MTF) of the MultiSpectral (MS) sensor. 
 
 Interface:
           I_Filtered = MTF(I_MS,sensor,ratio)

 Inputs:
           I_MS:           MS image;
           sensor:         String for type of sensor (e.g. 'WV2', 'IKONOS');
           ratio:          Scale ratio between MS and PAN.

 Outputs:
           I_Filtered:     Output filtered MS image.
 
 Notes:
     The bottleneck of this function is the function scipy.filters.correlate that gets the same results as in the MATLAB toolbox
     but it is very slow with respect to fftconvolve that instead gets slightly different results

 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.           
"""
from scipy import ndimage
import numpy as np
from .genMTF_pan import genMTF_pan


def MTF_pan(I_MS, sensor, ratio):
    h = genMTF_pan(ratio, sensor, I_MS.shape[2])

    I_MS_LP = np.zeros((I_MS.shape))
    for ii in range(I_MS.shape[2]):
        I_MS_LP[:, :, ii] = ndimage.filters.correlate(I_MS[:, :, ii], h, mode='nearest')
        ### This can speed-up the processing, but with slightly different results with respect to the MATLAB toolbox
        # hb = h[:,:,ii]
        # I_MS_LP[:,:,ii] = signal.fftconvolve(I_MS[:,:,ii],hb[::-1],mode='same')

    return np.double(I_MS_LP)
