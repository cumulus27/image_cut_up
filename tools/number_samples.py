#!/usr/bin/python2
# -*- coding: utf-8 -*

import os
import cv2
from ImagePartition import ImagePartition
from ImagePartition import MouldDetect
from fastrotate import mser

if __name__ == '__main__':
    # Input image
    decode = 1
    for i in range(932):
        bann = str(i + 1)
        while len(bann) < 6:
            bann = '0' + bann

        print(bann)

        src = "/home/ad/dataset/20171213/select/{}.jpg".format(bann)
        srcWrite = "/home/ad/dataset/20171213/selects/"

        if os.path.exists(src):
            RGB = cv2.imread(src)
        else:
            print('File not exists!!')
            continue

        srcPath = srcWrite

        filename = str(decode)
        while len(filename) < 6:
            filename = '0' + filename

        srcWrites = srcPath + filename + '.jpg'
        print(srcWrites)
        if not os.path.exists(srcPath):
            os.makedirs(srcPath)
        cv2.imwrite(srcWrites, RGB,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        decode += 1




    print('end!!!')
    cv2.waitKey(0)
    cv2.destroyAllWindows()