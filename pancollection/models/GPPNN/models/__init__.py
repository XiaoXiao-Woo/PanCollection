# -*- coding: utf-8 -*-

def get_sat_param(
        sat_str):
    if sat_str.lower()=='landsat8' or sat_str.lower()=='l8':
        ms_channels = 10
        pan_channels = 1
        scale = 2
        return ms_channels, pan_channels, scale
    elif sat_str.lower()=='quickbird' or sat_str.lower()=='qb':
        ms_channels = 4
        pan_channels = 1
        scale = 4
        return ms_channels, pan_channels, scale
    elif sat_str.lower()=='gf2' or sat_str.lower()=='gaofen2':
        ms_channels = 4
        pan_channels = 1
        scale = 4
        return ms_channels, pan_channels, scale
    else:
        output = None
        return output
    

    
