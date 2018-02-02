# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 09:16:50 2018

@author: zhaoy
"""


def bilinear_interp(x, x1, x2, y1, y2):
    assert(x2 > x1)
    assert(x >= x1)
    assert(x <= x2)

    dx12 = x2 - x1
    f1 = (x - x1) / dx12
    f2 = (x2 - x) / dx12

    y = y1 * f2 + y2 * f1

    # print f1, f2
    # print y

    return y


if __name__ == '__main__':
    x1 = 7.599999776175537e-07
    x2 = 1.699999984339229e-06
    x = 1e-6

    y1 = 0.9320723414421082
    y2 = 0.9421482086181641

    y = bilinear_interp(x, x1, x2, y1, y2)
    print 'y = ', y
