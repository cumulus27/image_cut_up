#!/usr/bin/python2
# -*- coding: utf-8 -*

import os
import numpy as np
import cv2
from pylab import *

import single_line
import FixLine


if __name__ == '__name__':

    for numb in range(915,933):
        bann = str(numb + 1)
        while len(bann) < 6:
            bann = '0' + bann
        # print(bann)
        src = "~/dataset/outzheng5/{}.jpg".format(bann)
        print (src)
        # src="./outzheng5/000009.jpg"
        A=cv2.imread(src)
        LAB=cv2.cvtColor(A,cv2.COLOR_BGR2LAB)
        V=LAB[:,:,0]
        eq1=cv2.equalizeHist(V)
        LAB[:,:,0]=eq1
        AAA=cv2.cvtColor(LAB, cv2.COLOR_LAB2BGR)
        gray=cv2.cvtColor(AAA,cv2.COLOR_BGR2GRAY)
        (w,h) = gray.shape
        gray = cv2.resize(gray, (h//2,w//2))
        x1, y1, x2, y2, wmid, hmid = single_line.separate(gray)
        if len(x1) > 0 and len(x2) > 0:
            x_left1 = min(x1)
            y_left1 = y1[np.where(x1 == x_left1)][0]
            x_right1 = max(x1)
            y_right1 = y1[np.where(x1 == x_right1)][0]
            x_left2 = min(x2)
            y_left2 = y2[np.where(x2 == x_left2)][0]
            x_right2 = max(x2)
            y_right2 = y2[np.where(x2 == x_right2)][0]

            xl_mid = (x_left1+x_left2)//2
            yl_mid = (y_left1+y_left2)//2
            xr_mid = (x_right1+x_right2)//2
            yr_mid = (y_right1+y_right2)//2

            cv2.line(A, (xl_mid, (yl_mid+hmid//2)*2), (xr_mid, (yr_mid+hmid//2)*2), (0,255,0))
            if xr_mid-xl_mid == 0:
                xr_mid = xl_mid+0.01
            k = float(yr_mid-yl_mid)/float(xr_mid-xl_mid)
            for j in range(1,h//2):
                for i in range(1,w//2):
                    if i == int((j-xl_mid)*k+yl_mid+hmid//2):
                        # print(i,j)
                        A[i*2,j*2] = (0,0,255)
            cv2.imshow('A', A)
            cv2.waitKey()
            single_line.extrct(x1, y1 ,wmid, hmid, gray)
            single_line.extrct(x2, y2, wmid, hmid, gray)
