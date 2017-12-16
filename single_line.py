# -*- coding: utf-8 -*

from __future__ import division
import os
import numpy as np
import cv2
from pylab import *
from matplotlib import pyplot as plt
from FixLine import FixLine


def separate(img=[]):
    rot_img = img
    mser = cv2.MSER_create(_min_area=40, _max_area=100)
    regions, boxes = mser.detectRegions(rot_img)

    wrange = 4
    hrange = 4
    if len(boxes) != 0:
        histnum_w = hist(boxes[:, 2])
        wmax_index = find(histnum_w[0][:] == max(histnum_w[0][:]))
        wmin = histnum_w[1][wmax_index] - wrange
        wmax = histnum_w[1][wmax_index] + wrange
        histnum_h = hist(boxes[:, 3])
        hmax_index = find(histnum_h[0][:] == max(histnum_h[0][:]))
        hmin = histnum_h[1][hmax_index] - hrange
        hmax = histnum_h[1][hmax_index] + hrange
        wmid = int((wmax[0] + wmin[0]) / 2) + 2
        hmid = int((hmax[0] + hmin[0]) / 2) + 3

        xn = []
        yn = []
        temp = np.zeros(img.shape)
        for box in boxes:
            x, y, w1, h1 = box
            if w1 < wmax[0] and w1 > wmin[0] and h1 > hmin[0] and h1 < hmax[0]:
                if x in xn and y in yn:
                    pass
                else:
                    xn.append(x)
                    yn.append(y)
                    temp[y:y + h1, x:x + w1] = 1
        temp = np.uint8(temp)
        temp = img * temp
        # cv2.imshow('tmp', temp)
        # cv2.waitKey()
        xn = np.array(xn)
        yn = np.array(yn)
        if size(xn):
            x1 = xn[np.where(yn < min(yn) + np.min(hmin))]
            y1 = yn[np.where(yn < min(yn) + np.min(hmin))]
            x2 = xn[np.where(yn > min(yn) + np.max(hmax))]
            y2 = yn[np.where(yn > min(yn) + np.max(hmax))]
            return x1, y1, x2, y2, wmid, hmid
        else:
            return [], [], [], [], 0, 0
            print("请手动")
    else:
        return [], [], [], [], 0, 0
        print("请手动")


# def extrct(xn, yn, wmid, hmid, str_gray):
#     if xn != []:
#         x_left = min(xn)
#         y_left = yn[np.where(xn == x_left)][0]
#         x_right = max(xn)
#         y_right = yn[np.where(xn == x_right)][0]
#         k = (y_right - y_left) / (x_right - x_left)
#         y_mid = (y_left + y_right) // 2
#         y_line = y_mid + hmid
#         (h, w) = str_gray.shape[:2]
#         # line = np.zeros((h, w))
#         # line[y_mid - 5:y_line + 5, :] = 1
#         # numin = line * str_gray
#         numin = np.uint8(str_gray)
#         angle = np.rad2deg(np.arctan(k))
#         center = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center, angle, 1.0)
#         h_new = int(w * fabs(sin(radians(angle))) + h * fabs(cos(radians(angle))))
#         w_new = int(h * fabs(sin(radians(angle))) + w * fabs(cos(radians(angle))))
#         rotated = cv2.warpAffine(numin, M, (w_new, h_new), borderValue=(0))
#
#         rotated = np.uint8(rotated)
#         cv2.imshow('rot', rotated)
#     else:
#         pass

def dline(A):
    # A = cv2.imread(src)
    LAB = cv2.cvtColor(A, cv2.COLOR_BGR2LAB)
    V = LAB[:, :, 0]
    eq1 = cv2.equalizeHist(V)
    LAB[:, :, 0] = eq1
    AAA = cv2.cvtColor(LAB, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(AAA, cv2.COLOR_BGR2GRAY)
    (w, h) = gray.shape
    gray = cv2.resize(gray, (h // 2, w // 2))
    x1, y1, x2, y2, wmid, hmid = separate(gray)
    if len(x1) > 0 and len(x2) > 0:
        x_left1 = min(x1)
        y_left1 = y1[np.where(x1 == x_left1)][0]
        x_right1 = max(x1)
        y_right1 = y1[np.where(x1 == x_right1)][0]
        x_left2 = min(x2)
        y_left2 = y2[np.where(x2 == x_left2)][0]
        x_right2 = max(x2)
        y_right2 = y2[np.where(x2 == x_right2)][0]

        xl_mid = (x_left1 + x_left2) // 2
        yl_mid = (y_left1 + y_left2) // 2
        xr_mid = (x_right1 + x_right2) // 2
        yr_mid = (y_right1 + y_right2) // 2

        if xr_mid - xl_mid == 0:
            xr_mid = xl_mid + 0.01
        k = float(yr_mid - yl_mid) / float(xr_mid - xl_mid)
        x = []
        y = []
        for j in range(1, h // 2):
            for i in range(1, w // 2):
                if i == int((j - xl_mid) * k + yl_mid + hmid // 2):
                    # print(i,j)
                    y.append(2 * i)
                    x.append(2 * j)
        # cv2.line(A, (x[1], y[1]), (x[-1], y[-1]), (0, 255, 0))
        # 最左边x[1],y[1]
        # 最右边x[-1], y[-1]
    # cv2.imshow('A', A)
    # cv2.waitKey()
    return x[1], y[1], x[-1], y[-1]

if __name__ == '__main__':

    for numb in range(137,933):
        bann = str(numb + 1)
        while len(bann) < 6:
            bann = '0' + bann
        # print(bann)
        src = "/home/ad/dataset/outzheng5/{}.jpg".format(bann)
        print (src)

        A=cv2.imread(src)
        x1, y1, x2, y2 = dline(A)
        An = A.copy()
        cv2.line(A, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # hand fix
        fix = FixLine(An, A)

        fix.fix_line(bann)

        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        dst, line1, line2 = fix.split_line(p1, p2)

        cv2.imshow('naive', An)
        cv2.imshow('fix', dst)
        cv2.imshow('line1', line1)
        cv2.imshow('line2', line2)
        cv2.waitKey()
        cv2.destroyAllWindows()






