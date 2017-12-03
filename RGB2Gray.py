#!/usr/bin/python2
# -*- coding: utf-8 -*

import os
import cv2
from ImagePartition import ImagePartition
from ImagePartition import MouldDetect
from fastrotate import mser

if __name__ == '__main__':
    # Input image
    for i in range(773,932):
        bann = str(i + 1)
        while len(bann) < 6:
            bann = '0' + bann

        print(bann)
        src = "./out1/{}.jpg".format(bann)
        if os.path.exists(src):
            RGB = cv2.imread(src)
        else:
            print('File not exists!!')
            continue

        RGB = cv2.imread(src)

        ms = mser(RGB)
        ms.cvt2gray()
        ms.graystretch()
        ms.kmeans()
        ms.mser()
        rot = ms.rotate()

        gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)

        if not os.path.exists("./gray/"):
            os.makedirs("./gray/")

        cv2.imwrite("./gray/{}.jpg".format(bann), gray,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])