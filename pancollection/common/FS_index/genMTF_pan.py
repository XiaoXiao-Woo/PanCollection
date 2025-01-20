# -*- coding: utf-8 -*-
import numpy as np
from .tools import fir_filter_wind, gaussian2d, kaiser2d


def genMTF_pan(ratio, sensor, nbands):
    N = 41

    if sensor == 'QB':
        GNyq = 0.15
    elif sensor == 'IKONOS':
        GNyq = 0.17
    elif sensor in ['GeoEye1', 'WV4']:
        GNyq = 0.16
    elif sensor == 'WV2':
        GNyq = 0.11
    elif sensor == 'WV3':
        GNyq = 0.14
    else:
        GNyq = 0.15

    """MTF"""
    fcut = 1 / ratio
    # h = np.zeros((N, N, nbands))
    alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(GNyq)))
    H = gaussian2d(N, alpha)
    Hd = H / np.max(H)
    w = kaiser2d(N, 0.5)
    h = np.real(fir_filter_wind(Hd, w))

    return h
