#!/usr/bin/python2
# -*- coding: utf-8 -*

import os
import numpy as np
import cv2


class FixLine(object):
    """Fix Line

    """
    def __init__(self, naive, img):
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

        while True:
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

    point1 = np.array([x1, y1])
    point2 = np.array([x2, y2])
    ll = np.sqrt(np.sum(np.square(point1-point2)))
    print(ll)

    ww = ll / 4.5
    wbias = int(ww // 2)
    print(wbias)

    rows, cols, channel = RGB.shape
    print((rows, cols))

    rowsn = wbias * 2
    colsn = int(ll) + 1

    yy = y2 - y1
    xx = x2 - x1

    nb1x = (yy / ll) * wbias
    nb1y = (-xx / ll) * wbias
    nb2x = (-yy / ll) * wbias
    nb2y = (xx / ll) * wbias

    nb1 = np.array([nb1x, nb1y])
    nb2 = np.array([nb2x, nb2y])
    print(nb1)
    print(nb2)

    f1 = point1+nb1
    f2 = point1+nb2
    f3 = point2+nb1
    print(f1)
    print(f2)
    print(f3)

    print(max(0,f1[0]))
    print(max(0,f1[1]))

    f11 = np.array([max(max(0, f1[0]), min(cols, f1[0])), max(max(0, f1[1]), min(rows, f1[1]))])
    f22 = np.array([max(max(0, f2[0]), min(cols, f2[0])), max(max(0, f2[1]), min(rows, f2[1]))])
    f33 = np.array([max(max(0, f3[0]), min(cols, f3[0])), max(max(0, f3[1]), min(rows, f3[1]))])
    print(f11)
    fs11 = np.array([max(0, -f1[0]), max(0, -f1[1])])
    fs22 = np.array([max(0, -f2[0]), max(0, -f2[1])])
    fs33 = np.array([max(0, -f3[0]), max(0, -f3[1])])
    print(fs11)


    # pts1 = np.float32([point1-nb1, point1-nb2, point2-nb1])
    pts1 = np.float32([[13,272], [50, 334], [348, 18]])
    # pts2 = np.float32([[0, 0], [rowsn, 0], [0, colsn]])
    pts2 = np.float32([[0, 0], [0, rowsn], [colsn, 0]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(RGB, M, (colsn, rowsn))

    cv2.imshow('fix',dst)
    cv2.waitKey()


