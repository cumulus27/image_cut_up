#!/usr/bin/python2
# -*- coding: utf-8 -*


import cv2
from pylab import *
import copy
import os
class mser:

    def __init__(self, img, outpath=[]):
        # self.src = src
        # self.outpath = outpath
        # self.img = cv2.imread(self.src)
        self.img0 = img
        hh = img.shape[0]
        ww = img.shape[1]
        self.img = cv2.resize(self.img0, (ww / 2, hh / 2), interpolation=cv2.INTER_AREA)
        self.gray = []
        self.str_gray = []
        self.regions = []
        self.boxes = []
        self.inverse_regions = []
        self.inverse_boxes = []
        self.index0 = []
        self.index1 = []
        self.res2 = []

    def cvt2gray(self,img=[]):
        if img == []:
            img = self.img
            LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            L = LAB[:, :, 0]
            # cv2.imshow('LAB', LAB)
            eq1 = cv2.equalizeHist(L)
            LAB[:, :, 0] = eq1
            # cv2.imshow('LAB1', LAB)
            BRG = cv2.cvtColor(LAB, cv2.COLOR_LAB2BGR)
            self.gray=cv2.cvtColor(BRG, cv2.COLOR_BGR2GRAY)

        else:
            LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            L = LAB[:, :, 0]
            eq1 = cv2.equalizeHist(L)
            LAB[:, :, 0] = eq1
            BRG = cv2.cvtColor(LAB, cv2.COLOR_LAB2BGR)
            return cv2.cvtColor(BRG, cv2.COLOR_BGR2GRAY)

    def graystretch(self, gray=[]):
        if gray == []:
            gray=self.gray
            histeq = cv2.equalizeHist(gray)
            self.str_gray = histeq
        else:
            histeq = cv2.equalizeHist(gray)
            return histeq

    def kmeans(self):
        temp = self.str_gray.reshape(-1, 1)
        temp = np.float32(temp)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        ret, label, center = cv2.kmeans(temp, 7, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        self.res2 = res.reshape(self.str_gray.shape)

    def mser(self):
        mser_init = cv2.MSER_create(_min_area=30, _max_area=120)
        self.regions, self.boxes = mser_init.detectRegions(self.res2)

    def rot_rect(self, cosv, tanv, lx, ly):
        tmp = np.zeros((lx, ly))

        if cosv != 0:
            for w in np.arange(np.round(-abs(20 / cosv), 2), np.round(abs(20 / cosv), 2)):
                for i in range(0, lx):
                    for j in range(0, ly):
                        if (i - lx // 2) == int((-tanv * (j - ly // 2)) + w):
                            tmp[i, j] = 1
        else:
            for i in range(0, lx):
                for j in range(0, ly):
                    if j > ly // 2 - 21 and j < ly // 2 + 21:
                        tmp[i, j] = 1
        return tmp

    def rotate(self):
        # 宽度和高度直方图
        w_floor = 10
        h_floor = 7
        wrange = 5
        hrange = 7
        (h, w) = self.img.shape[:2]
        center = (w // 2, h // 2)
        # 用宽度和高度约束
        temp = np.zeros(self.res2.shape)
        histnum_w = hist(self.boxes[:, 2])
        wmax_index = find(histnum_w[0][:] == max(histnum_w[0][:]))
        wmin = histnum_w[1][wmax_index] - wrange
        wmax = histnum_w[1][wmax_index] + wrange

        histnum_h = hist(self.boxes[:, 3])
        hmax_index = find(histnum_h[0][:] == max(histnum_h[0][:]))
        hmin = histnum_h[1][hmax_index] - hrange
        hmax = histnum_h[1][hmax_index] + hrange
        # 针对麻点图
        if len(self.boxes[:, 0]) > 100:
            wmin[0] = w_floor
            wmax[0] = w_floor+9
            hmin[0] = h_floor
            hmax[0] = h_floor+7
        # 筛选框取区域
        for box in self.boxes:
            x, y, w1, h1 = box
            if w1 < wmax[0] and w1 > wmin[0] and h1 > hmin[0]  and h1 < hmax[0]:
                temp[y:y+h1, x:x+w1] = 1
        temp = np.uint8(temp)
        temp = np.multiply(self.res2, temp)
        for i in range(0, h1):
            for j in range(0, w1):
                if temp[i, j] < 130:
                    temp[i, j] = 0
        max_pixel = 0
        max_angle = 0
        hist1 = cv2.equalizeHist(temp)

        # 用叠影法求最大方向
        shadow = np.zeros((h, w))
        global bool_best
        swidth=18
        bool_best = np.zeros((h, w))
        if w > h:
            shadow[h // 2 - swidth:h // 2 + swidth, :] = 1
            flag_0 = 0
        else:
            shadow[:, w//2-swidth:w//2+swidth] = 1
            flag_0 = 1
        for i in np.arange(0, 180, 0.5):
            tmp1 = copy.deepcopy(temp)
            boole = copy.deepcopy(shadow)
            M_s = cv2.getRotationMatrix2D(center, i, 1.0)
            rot_boole = cv2.warpAffine(boole, M_s, (w,h),borderValue=(0))
            rot_boole = np.uint8(rot_boole)
            mult = rot_boole*tmp1
            pixel_sum = np.sum(mult)
            if pixel_sum > max_pixel:
                bool_best = rot_boole
                max_pixel = pixel_sum
                max_angle = i
        if flag_0 == 0:
            angle = max_angle
        else:
            angle = max_angle-90
        bool_best=np.uint8(bool_best)
        h_new = int(w * fabs(sin(radians(angle))) + h * fabs(cos(radians(angle))))
        w_new = int(h * fabs(sin(radians(angle))) + w * fabs(cos(radians(angle))))
        center = (w, h)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        M[0, 2] += (w_new - w)
        M[1, 2] += (h_new - h)
        rotated = cv2.warpAffine(self.img0, M, (w_new*2, h_new*2), borderValue=(0))
        rotated = np.uint8(rotated)
        srot = np.zeros((h_new*2, w_new*2,3))
        srot[h_new - 24*2:h_new + 24*2, :,:] = 1
        srot=np.uint8(srot)
        self.rotated2=srot*rotated
        cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print("[INFO] angle: {:.3f}".format(angle))
        # cv2.imshow(self.src+'num', self.rotated2)
        return self.rotated2

    def separate(self,img=[]):
        if img==[]:
            rot_img = self.rotated2
        else:
            rot_img = img
        gray = self.cvt2gray(rot_img)
        str_gray = self.graystretch(gray)
        mser = cv2.MSER_create(_min_area=40, _max_area=90)
        regions, boxes = mser.detectRegions(str_gray)

        wrange = 2
        hrange = 2
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
        for box in boxes:
            x, y, w1, h1 = box
            if w1 < wmax[0] and w1 > wmin[0] and h1 > hmin[0] and h1 < hmax[0]:
                if x in xn and y in yn:
                    pass
                else:
                    xn.append(x)
                    yn.append(y)

        xn = np.array(xn)
        yn = np.array(yn)
        x1 = xn[np.where(yn < min(yn) + hmin)]
        y1 = yn[np.where(yn < min(yn) + hmin)]
        x2 = xn[np.where(yn > min(yn) + hmax)]
        y2 = yn[np.where(yn > min(yn) + hmax)]

        return x1, y1, x2, y2, wmid, hmid,str_gray

    def extrct(self,xn, yn, wmid ,hmid,str_gray):
        x_left = min(xn)
        y_left = yn[np.where(xn == x_left)][0]
        x_right = max(xn)
        y_right = yn[np.where(xn == x_right)][0]
        k = (y_right - y_left) / (x_right - x_left)
        y_mid = (y_left + y_right) // 2
        y_line = y_mid + hmid
        (h, w) = self.rotated2.shape[:2]
        line = np.zeros((h, w))
        line[y_mid - 3:y_line + 2, :] = 1
        numin = line * str_gray
        numin = np.uint8(numin)
        angle = rad2deg(arctan(k))
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(numin, M, (w, h), borderValue=(0))
        rotated = line * rotated
        rotated = np.uint8(rotated)
        #如何找到合适的阈值？直方图是否可行
        thresh = 210
        ret, rotated = cv2.threshold(rotated, thresh, 255, cv2.THRESH_BINARY)
        cv2.imshow('rot2', rotated)
        column_sum = np.sum(rotated, 0)
        xline = np.zeros((1, 0))
        for i in range(0, len(column_sum) - 1):
            if column_sum[i] < 50 and column_sum[i + 1]:
                xline = np.append(xline, (i + 1))
        xline = np.append(xline, xline[-1] + wmid)
        xline = np.uint8(xline)

        #示范，其实只要返回xline就行。
        for i in range(0, len(xline) - 1):
            temp = zeros((h, w))
            tmp = zeros((h, w, 3))
            temp[y_mid - 3:y_line + 2, xline[i]:xline[i + 1]] = 1
            tmp[:, :, 0] = temp
            tmp[:, :, 1] = temp
            tmp[:, :, 2] = temp
            temp = temp * rotated
            temp = np.uint8(temp)
            tmp = np.uint8(tmp)
            cv2.imshow('gray1', temp)
            cv2.waitKey()
        return xline

if __name__ == "__main__":
    # 这段是文件遍历
    # FindPath = '/home/yxt/py_coding/ciga_rec/out1/out1/'
    # SavePath = '/home/yxt/py_coding/ciga_rec/out1/extract/'
    # FileNames = os.listdir(FindPath)
    # for file_name in FileNames:
    #     src = os.path.join(FindPath, file_name)
        # outpath = os.path.join(SavePath, file_name)
        # print(src)

    #这段是指定的图
        bann = '000889'
        src = "./out1/{}.jpg".format(bann)
        img = cv2.imread(src)
        ms = mser(img)
        ms.cvt2gray()
        ms.graystretch()
        ms.kmeans()
        ms.mser()
        rot=ms.rotate()
        # x1,y1,x2,y2,wmid,hmid,str_gray=ms.separate()
        # ms.extrct(x1,y1,wmid,hmid,str_gray)
        # ms.extrct(x2, y2, wmid, hmid, str_gray)
        cv2.imshow('rot', rot)
        cv2.waitKey()

