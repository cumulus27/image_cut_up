#!/usr/bin/python2
# -*- coding: utf-8 -*

import os
import cv2
from ImagePartition import ImagePartition
from ImagePartition import MouldDetect
from fastrotate import mser

if __name__ == '__main__':
    # Input image
    for i in range(932):
        bann = str(i + 1)
        while len(bann) < 6:
            bann = '0' + bann

        print(bann)
        src = "./out/{}.jpg".format(bann)
        if os.path.exists(src):
            RGB = cv2.imread(src)
        else:
            print('File not exists!!')
            continue

        RGB = cv2.imread(src)

        # src = "/home/yxt/py_coding/ciga_rec/out1/out1/000723.jpg"
        cv2.destroyAllWindows()
        ms = mser(RGB)
        ms.cvt2gray()
        ms.graystretch()
        ms.kmeans()
        ms.mser()
        rot = ms.rotate()

        # make image patition
        RGB = rot
        partition = ImagePartition(RGB)
        partition.partition_operate()
        src = "./result/all/"
        partition.write_image_all(src, bann)


    print('end!!!')
    cv2.waitKey(0)
    cv2.destroyAllWindows()