#!/usr/bin/python2
# -*- coding: utf-8 -*

import os
import cv2
from ImagePartition import ImagePartition
from ImagePartition import MouldDetect
from fastrotate import mser

if __name__ == '__main__':
    # Input image
    for i in range(70,932):
        bann = str(i + 1)
        while len(bann) < 6:
            bann = '0' + bann

        print(bann)
        # src = "./out/{}.jpg".format(bann)
        srcper = "/home/ad/dataset/outzheng5_rot/{}.jpg".format(bann)
        src = "/home/ad/dataset/outzheng5/{}.jpg".format(bann)
        if os.path.exists(srcper):
            RGB = cv2.imread(srcper)
        elif os.path.exists(src) and i > 392:
            RGB = cv2.imread(src)
        else:
            print('File not exists!!')
            continue

        # RGB = cv2.imread(src)

        if RGB.shape[0] > 200:
            print("Can't use!!")
            continue
        # print(RGB.shape)
        # cv2.imshow('now', RGB)

        # cv2.waitKey()


        # # src = "/home/yxt/py_coding/ciga_rec/out1/out1/000723.jpg"
        # cv2.destroyAllWindows()
        # ms = mser(RGB)
        # ms.cvt2gray()
        # ms.graystretch()
        # ms.kmeans()
        # ms.mser()
        # rot = ms.rotate()
        #
        # # make image patition
        # RGB = rot
        cv2.destroyAllWindows()
        partition = ImagePartition(RGB)
        partition.partition_operate()
        src = "/home/ad/dataset/result5_3p/all_new/"
        partition.write_image_all(src, bann)


    print('end!!!')
    cv2.waitKey(0)
    cv2.destroyAllWindows()