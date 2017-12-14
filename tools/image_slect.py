#!/usr/bin/python2
# -*- coding: utf-8 -*

import os
import sys
import termios
import cv2
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
               57: 'pass', 28: 'pass', 1: 'exit', 83: 'del', 111: 'del', 78: 'err', 13: 'err',
               109: 'skip', }
    srcWrite = '/home/ad/dataset/20171213/'
    # sn = [0]*10
    # sn = getCurrentNumber(srcWrite)

    for i in range(123,932):
        bann = str(i + 1)
        while len(bann) < 6:
            bann = '0' + bann

        # srcper = "/home/ad/dataset/outzheng5_rot/{}.jpg".format(bann)
        # src = "/home/ad/dataset/outzheng5/{}.jpg".format(bann)
        # if os.path.exists(srcper):
        #     RGB = cv2.imread(srcper)
        # elif os.path.exists(src):
        #     RGB = cv2.imread(src)
        # else:
        #     print('File not exists!!')
        #     continue
        src = "/home/ad/dataset/20171213/test/{}.jpg".format(bann)
        if os.path.exists(src):
            RGB = cv2.imread(src)
        else:
            print('File not exists!!')
            continue

        # if RGB.shape[0] > 200:
        #     print("Can't use!!")
        #     continue

        src0 = '/home/ad/dataset/outlines5_2/'
        filename1 = bann + '_line1'
        src1 = src0 + filename1 + '.jpg'
        filename2 = bann + '_line2'
        src2 = src0 + filename2 + '.jpg'

        cv2.destroyAllWindows()
        cv2.imshow('naive{}'.format(bann), RGB)
        if os.path.exists(src1):
            RGB1 = cv2.imread(src1)
            cv2.imshow(filename1, RGB1)
        else:
            pass

        if os.path.exists(src2):
            RGB2 = cv2.imread(src2)
            cv2.imshow(filename2, RGB2)
        else:
            pass


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

        if decode == 'skip':
            pass
        elif decode == 'del':
            # os.remove(src)
            continue
        elif decode == 'pass':
            srcPath = srcWrite + 'select/'
            srcWrites = srcPath + bann + '.jpg'
            print(srcWrites)
            if not os.path.exists(srcPath):
                os.makedirs(srcPath)
            cv2.imwrite(srcWrites, RGB,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # os.remove(src)
        elif decode == 'err':
            srcPath = srcWrite + 'select_plus/'
            srcWrites = srcPath + bann + '.jpg'
            print(srcWrites)
            if not os.path.exists(srcPath):
                os.makedirs(srcPath)
            cv2.imwrite(srcWrites, RGB,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # os.remove(src)
        else:
            # sn[decode] += 1
            # snStr = str(sn[decode])
            # while len(snStr) < 8:
            #     snStr = '0' + snStr

            srcPath = srcWrite + '{}/'.format(decode)
            srcWrites = srcPath + bann + '.jpg'
            print(srcWrites)
            if not os.path.exists(srcPath):
                os.makedirs(srcPath)
            cv2.imwrite(srcWrites, RGB,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # os.remove(src)