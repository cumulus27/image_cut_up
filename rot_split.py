#!/usr/bin/python2
# -*- coding: utf-8 -*

import os
import cv2
from ImagePartition import ImagePartition
from ImagePartition import MouldDetect
from fastrotate import mser

if __name__ == '__main__':
    # Input image
    bann = '000029'
    src = "./out1/{}.jpg".format(bann)
    if os.path.exists(src):
        RGB = cv2.imread(src)
    else:
        print('File not exists!!')


    # src = "/home/yxt/py_coding/ciga_rec/out1/out1/000723.jpg"
    ms = mser(RGB)
    ms.cvt2gray()
    ms.graystretch()
    ms.kmeans()
    ms.mser()
    rot = ms.rotate()
    cv2.imshow('rot result', rot)

    # make image patition
    RGB = rot
    partition = ImagePartition(RGB)
    partition.partition_operate()
    partition.show_image_result()
    partition.show_md_result()

    cv2.waitKey(0)
    cv2.destroyAllWindows()