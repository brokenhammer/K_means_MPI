#! /usr/bin/env python
# -*-encoding:utf-8-*-

__author__ = "xswei"

import numpy as np
def distance(point1,point2):
    # here we use Euclidean distance
    assert(point1.shape == point2.shape)
    assert(len(point1.shape) <= 2)
    if len(point1.shape) == 1:
        d2 = 0
        for dim in range(len(point1)):
            d2 += (point1[dim] - point2[dim])^2
        return np.sqrt(d2)
    else:
        d = np.zeros(point1.shape[0])
        for p in range(len(point1.shape[0])):
            d[p] = distance(point1[p],point2[p])
        return d