#!/usr/bin/python2
# -*- coding: utf-8 -*

import os
import sys
import termios
import cv2
# from ImagePartition import ImagePartition
# from ImagePartition import MouldDetect
# from select import select

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
    keyDict = {48: 0, 49: 1, 50: 2, 51: 3, 52: 4, 53: 5, 54: 6, 55: 7, 56: 8, 57: 9, 32: 'pass',
               13: 'pass', 27: 'exit'}
    srcWrite = '/home/py/dataset/shuzi_copy/'
    # sn = [0]*10
    # sn = getCurrentNumber(srcWrite)

    for i in range(330,630):
        bann = str(i + 1)
        while len(bann) < 6:
            bann = '0' + bann

        src0 = '/home/py/dataset/shuzi_copy/naive/'

        for j in [1]:
            for k in range(10,50):
                filename = bann + '_{}'.format(k)
                src = src0 + filename + '.jpg'

                if os.path.exists(src):
                    RGB = cv2.imread(src)
                else:
                    continue

                cv2.destroyAllWindows()

                while True:
                    cv2.imshow(filename, RGB)
                    value = cv2.waitKey(5) & 0xFF
                    if value != 255:
                        print(value)
                    try:
                        decode = keyDict[value]
                        print("Key: %s Mean: %s" % (value, decode) )
                        break
                    except:
                        decode = None

                if decode == 'exit':
                    sys.exit(0)

                if decode == 'pass':
                    srcPath = srcWrite + 'pass/'
                    srcWrites = srcPath + filename + '.jpg'
                    print(srcWrites)
                    if not os.path.exists(srcPath):
                        os.makedirs(srcPath)
                    cv2.imwrite(srcWrites, RGB,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    os.remove(src)
                else:
                    srcPath = srcWrite + '{}/'.format(decode)
                    srcWrites = srcPath + filename + '.jpg'
                    print(srcWrites)
                    if not os.path.exists(srcPath):
                        os.makedirs(srcPath)
                    cv2.imwrite(srcWrites, RGB,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    os.remove(src)
