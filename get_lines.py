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
        # src = "./out1/{}.jpg".format(bann)
        src = "/home/ad/dataset/outzheng5/{}.jpg".format(bann)
        if os.path.exists(src):
            RGB = cv2.imread(src)
        else:
            print('File not exists!!')
            continue

        RGB = cv2.imread(src)

        if RGB.shape[0] > 200:
            print("Can't use!!")
            continue

        # src = "/home/yxt/py_coding/ciga_rec/out1/out1/000723.jpg"
        cv2.destroyAllWindows()
        # ms = mser(RGB)
        # ms.cvt2gray()
        # ms.graystretch()
        # ms.kmeans()
        # ms.mser()
        # rot = ms.rotate()

        # make image patition
        # RGB = rot
        partition = ImagePartition(RGB)
        cv2.destroyAllWindows()
        # partition.partition_operate()
        g1, g2 = partition.user_edit()

        # srcw = "./lines/"
        srcw = "/home/ad/dataset/outlines5/".format(bann)
        if not os.path.exists(srcw):
            os.makedirs(srcw)
        cv2.imwrite(srcw + "{}_line{}.jpg".format(bann, 1), g1,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        cv2.imwrite(srcw + "{}_line{}.jpg".format(bann, 2), g2,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])




    print('end!!!')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

