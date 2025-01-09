# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

"""
 Description: 
           Generate a bank of filters shaped on the MTF of the sensor. Each filter
           corresponds to a band acquired by the sensor. 
 
 Interface:
           h = genMTF(ratio, sensor, nbands)

 Inputs:
           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value.
           sensor:             String for type of sensor (e.g. 'WV2','IKONOS');
           nbands:             Number of spectral bands.

 Outputs:
           h:                  Gaussian filter mimicking the MTF of the MS sensor.
 
 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.         
"""

import numpy as np
from .tools import fir_filter_wind, gaussian2d, kaiser2d

def genMTF(ratio, sensor, nbands):
    
    N = 41
        
    if sensor=='QB':
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='IKONOS':
        GNyq = np.asarray([0.26,0.28,0.29,0.28],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='GeoEye1' or sensor=='WV4':
        GNyq = np.asarray([0.23,0.23,0.23,0.23],dtype='float32')    # Band Order: B,G,R,NIR
    elif sensor=='WV2':
        GNyq = [0.35*np.ones(nbands),0.27]
    elif sensor=='WV3':
        GNyq = np.asarray([0.325,0.355,0.360,0.350,0.365,0.360,0.335,0.315],dtype='float32') 
    else:
        GNyq = 0.3 * np.ones(nbands)
        
    """MTF"""
    h = np.zeros((N, N, nbands))

    fcut = 1/ratio

    h = np.zeros((N,N,nbands))
    for ii in range(nbands):
        alpha = np.sqrt(((N-1)*(fcut/2))**2/(-2*np.log(GNyq[ii])))
        H=gaussian2d(N,alpha)
        Hd=H/np.max(H)
        w=kaiser2d(N,0.5)
        h[:,:,ii] = np.real(fir_filter_wind(Hd,w))
        
    return h