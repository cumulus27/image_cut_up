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

        print((self.x1, self.y1))
        print((self.x2, self.y2))

        return self.x1, self.y1, self.x2, self.y2

    def split_line(self, p1, p2, ckb=4.5):

        bias = 4

        if self.x1 == -1 and self.x2 == -1:

            print('auto split!')
            point1 = p1
            point2 = p2

            print(p1)
            print(p2)
            ll = np.sqrt(np.sum(np.square(point1 - point2)))
            print(ll)

            wbias = 64 - bias
            print(wbias)

            yy = p2[1] - p1[1]
            xx = p2[0] - p1[0]

        else:
            point1 = np.array([self.x1, self.y1])
            point2 = np.array([self.x2, self.y2])

            ll = np.sqrt(np.sum(np.square(point1 - point2)))
            print(ll)
            ww = ll / ckb
            wbias = int(ww // 2)
            print(wbias)

            yy = self.y2 - self.y1
            xx = self.x2 - self.x1



        rows, cols, channel = self.naive_img.shape
        print((rows, cols))

        rowsn = wbias * 2
        colsn = int(ll) + 1

        print('yy and xx:')
        print(yy)
        print(xx)

        nb1x = (yy / ll) * wbias
        nb1y = (-xx / ll) * wbias
        nb2x = (-yy / ll) * wbias
        nb2y = (xx / ll) * wbias

        nb1 = np.array([nb1x, nb1y])
        nb2 = np.array([nb2x, nb2y])
        print(nb1)
        print(nb2)

        # f1 = point1+nb1
        # f2 = point1+nb2
        # f3 = point2+nb1
        # print(f1)
        # print(f2)
        # print(f3)
        #
        # print(max(0,f1[0]))
        # print(max(0,f1[1]))
        #
        # f11 = np.array([max(max(0, f1[0]), min(cols, f1[0])), max(max(0, f1[1]), min(rows, f1[1]))])
        # f22 = np.array([max(max(0, f2[0]), min(cols, f2[0])), max(max(0, f2[1]), min(rows, f2[1]))])
        # f33 = np.array([max(max(0, f3[0]), min(cols, f3[0])), max(max(0, f3[1]), min(rows, f3[1]))])
        # print(f11)
        # fs11 = np.array([max(0, -f1[0]), max(0, -f1[1])])
        # fs22 = np.array([max(0, -f2[0]), max(0, -f2[1])])
        # fs33 = np.array([max(0, -f3[0]), max(0, -f3[1])])
        # print(fs11)

        f11 = point1
        f22 = point2
        f33 = (point1 + point2) // 2 + nb1
        if f33[0] < 0 and f33[1] < 0:
            pass
        elif f33[0] < 0:
            pass
        elif f33[1] < 0:
            pass

        print(f11)
        print(f22)
        print(f33)

        print(colsn)
        print(rowsn)

        fs11 = np.array([0, wbias])
        fs22 = np.array([colsn, wbias])
        fs33 = np.array([colsn // 2, 0])
        print(fs11)
        print(fs22)
        print(fs33)

        pts1 = np.float32([f11, f22, f33])
        pts2 = np.float32([fs11, fs22, fs33])

        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(self.naive_img, M, (colsn, rowsn))


        line1 = dst[:wbias + bias, :]
        line2 = dst[wbias - bias:, :]
        print(line1.shape)
        print(line2.shape)
        return dst, line1, line2

if __name__ == '__main__':

    bann = '000021'
    src = "/home/ad/dataset/outzheng5/{}.jpg".format(bann)
    if os.path.exists(src):
        RGB = cv2.imread(src)
    else:
        print('File not exists!!')

    fix = FixLine(RGB, RGB)

    fix.fix_line(bann)
    p1 = p2 = np.array([-1, -1])
    dst, line1, line2 = fix.split_line(p1, p2)


    cv2.imshow('naive', RGB)
    cv2.imshow('fix',dst)
    cv2.imshow('line1',line1)
    cv2.imshow('line2',line2)
    cv2.waitKey()




