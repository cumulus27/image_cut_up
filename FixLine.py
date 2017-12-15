#!/usr/bin/python2
# -*- coding: utf-8 -*

import os
import numpy as np
import cv2


class FixLine(object):
    """Fix Line

    """
    def __init__(self, naive, img):
        self.drawing = False
        self.x1, self.y1 = -1, -1
        self.x2, self.y2 = -1, -1
        self.naive_img = naive
        self.img = img
        self.draw = False

    def draw_line(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw = True
            self.img = self.naive_img.copy()
            self.x1, self.y1 = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.draw:
                self.img = self.naive_img.copy()
                cv2.line(self.img, (self.x1, self.y1), (x, y), (0, 255, 0), 4)

        elif event == cv2.EVENT_LBUTTONUP:
            self.draw = False
            self.x2, self.y2 = x, y
            self.img = self.naive_img.copy()
            cv2.line(self.img, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 4)

    def fix_line(self, filename):

        cv2.namedWindow(filename)
        cv2.setMouseCallback(filename, self.draw_line)

        while (1):
            cv2.imshow(filename, self.img)
            k = cv2.waitKey(2) & 0xFF
            # print(k)
            if k == 13:
                break
        cv2.destroyAllWindows()
        return self.x1, self.y1, self.x2, self.y2

if __name__ == '__main__':
    bann = '000029'
    src = "./out1/{}.jpg".format(bann)
    if os.path.exists(src):
        RGB = cv2.imread(src)
    else:
        print('File not exists!!')

    fix = FixLine(RGB, RGB)

    x1, y1, x2, y2 = fix.fix_line(bann)

    print((x1,y1))
    print((x2,y2))


