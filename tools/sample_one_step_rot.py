#!/usr/bin/python2
# -*- coding: utf-8 -*

import os
import sys
import termios
import cv2
import numpy as np
from pylab import *
from matplotlib import pyplot as plt
import copy
# from ImagePartition import ImagePartition
# from ImagePartition import MouldDetect
from evdev import InputDevice
from select import select

def detectInputKey():
    dev = InputDevice('/dev/input/event4')
    select([dev], [], [])
    for event in dev.read():
        if event.value == 0  and event.code != 0:
            # print("Key: %s Status: %s" % (event.code, "pressed" if event.value else "release"))
            return event.code

def getCurrentNumber(srcWrite):
    sn = [0] * 10
    for i in range(10):
        for j in range(10000000):
            snStr = str(j+1)
            while len(snStr) < 8:
                snStr = '0' + snStr
            src = srcWrite + '{}/'.format(i) + snStr + '.jpg'
            if os.path.exists(src):
                continue
            else:
                sn[i] = j
                break

    print(sn)
    return sn



if __name__ == '__main__':
    # get image
    keyDict = {11: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 82: 0,
               79: 1, 80: 2, 81: 3, 75: 4, 76: 5, 77: 6, 71: 7, 72: 8, 73: 9, 96: 'pass',
               57: 'pass', 28: 'pass', 1: 'exit', 83: 'del'}
    srcWrite = '/home/ad/dataset/outzheng5_rot/'
    # sn = [0]*10
    # sn = getCurrentNumber(srcWrite)

    for i in range(932):
        bann = str(i + 1)
        while len(bann) < 6:
            bann = '0' + bann
        src0 = '/home/ad/dataset/outzheng5/'

        filename = bann
        src = src0 + filename + '.jpg'

        if os.path.exists(src):
            RGB = cv2.imread(src)
        else:
            break

        if RGB.shape[0] > 200:
            continue

        cv2.destroyAllWindows()
        cv2.imshow(filename, RGB)
        cv2.waitKey()

        while True:
            value = detectInputKey()
            try:
                decode = keyDict[value]
                print("Key: %s Mean: %s" % (value, decode))
                break
            except:
                decode = None

        if decode == 'exit':
            sys.exit(0)

        if decode == 'del':
            # os.remove(src)
            continue
        elif decode == 'pass':
            srcPath = srcWrite
            srcWrites = srcPath + filename + '.jpg'
            print(srcWrites)
            if not os.path.exists(srcPath):
                os.makedirs(srcPath)

            (h, w) = RGB.shape[:2]
            angle = 180
            center = (w // 2, h // 2)
            h_new_0 = int(RGB.shape[1] * fabs(sin(radians(angle))) + RGB.shape[0] * fabs(cos(radians(angle))))
            w_new_0 = int(RGB.shape[0] * fabs(sin(radians(angle))) + RGB.shape[1] * fabs(cos(radians(angle))))
            M_0 = cv2.getRotationMatrix2D(center, -angle, 1.0)
            RGB = cv2.warpAffine(RGB, M_0, (w_new_0, h_new_0), borderValue=(0))
            # RGB = np.uint8(RGB)
            RGB = np.uint8(RGB)

            cv2.imwrite(srcWrites, RGB,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # os.remove(src)
        else:
            pass

