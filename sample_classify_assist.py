#!/usr/bin/python2
# -*- coding: utf-8 -*

import os
import sys
import termios
import cv2
from ImagePartition import ImagePartition
from ImagePartition import MouldDetect
from evdev import InputDevice
from select import select

def detectInputKey():
    dev = InputDevice('/dev/input/event4')
    select([dev], [], [])
    for event in dev.read():
        if event.value == 1  and event.code != 0:
            print("Key: %s Status: %s" % (event.code, "pressed" if event.value else "release"))
            return event.code


if __name__ == '__main__':
    # get image
    keyDict = {11: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 82: 0,
               79: 1, 80: 2, 81: 3, 75: 4, 76: 5, 77: 6, 71: 7, 72: 8, 73: 9, 96: 'pass',
               57: 'pass', 28: 'pass', 1: 'exit'}
    srcWrite = '/home/ad/dataset/samples/'
    sn = [0]*10

    for i in range(932):
        bann = str(i + 1)
        while len(bann) < 6:
            bann = '0' + bann

        src0 = '/home/ad/dataset/result555/all_new/'

        for j in [1,2]:
            for k in range(1,30):
                filename = bann + '_{}sigle{}'.format(j, k)
                src = src0 + filename + '.jpg'

                if os.path.exists(src):
                    RGB = cv2.imread(src)
                else:
                    print('one line end.')
                    break

                cv2.destroyAllWindows()
                cv2.imshow(filename, RGB)
                cv2.waitKey()

                while True:
                    value = detectInputKey()

                    try:
                        decode = keyDict[value]
                        print(decode)
                        break
                    except:
                        decode = None

                if decode == 'exit':
                    sys.exit(0)

                if decode == 'pass':
                    os.remove(src)
                    continue
                else:
                    sn[decode] += 1
                    snStr = str(sn[decode])
                    while len(snStr) < 8:
                        snStr = '0' + snStr

                    srcPath = srcWrite + '{}/'.format(decode)
                    srcWrites = srcPath + snStr + '.jpg'
                    print(srcWrites)
                    if not os.path.exists(srcPath):
                        os.makedirs(srcPath)
                    cv2.imwrite(srcWrites, RGB,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    os.remove(src)

