# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 22:00:55 2024

@author: superiordrax
"""

import numpy as np
import re
import scipy.sparse as sp
from math import radians, cos, sin, asin, sqrt
deg=180.0/np.pi
rad=np.pi/180.0

def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(0))
    print(d)
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()

def readfile(filename):
    data=np.zeros([50,10])
    with open(filename,'r') as f:
        for i,line in enumerate(f.readlines()):
            if i==0:
                continue
            else:
                data[i-1,:]=re.split('\s+',line.strip())[1:]
        data[data>=9000]=np.nan
        return data

def writefile(data,filename):
    with open(filename,'w') as f:
        f.write('stn_id       p       t2      rh2      spd      dir      vis    rain6   rain24   tmax24   tmin24 \n')
        for i in range(50):
            write_data=data[i,:].squeeze()
            data_format=' {:05d}{:9.2f}{:9.2f}{:9.2f}{:9.2f}{:9.2f}{:9.2f}{:9.2f}{:9.2f}{:9.2f}{:9.2f} \n'
            f.write(data_format.format(i+1,write_data[0],write_data[1],write_data[2],write_data[3],write_data[4],write_data[5],write_data[6],write_data[7],write_data[8],write_data[9]))

def spdtouv(spd,wdir):
    u=-spd*np.sin(wdir*rad)
    v=-spd*np.cos(wdir*rad)
    return u,v

def uvtospd(u,v):
    spd=np.sqrt(u**2+v**2)
    wdir=180.0+np.arctan2(u,v)*deg
    return spd,wdir

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r * 1000



if __name__=='__main__':
    pass
    print(haversine(21,106,21.0625,106.0625))
    # a=readfile(r'./OBS_2023010100.000')
    # a[np.isnan(a)]=9999
    # writefile(a,'test.txt')
