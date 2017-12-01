#!/usr/bin/python2
# -*- coding: utf-8 -*

import os
import cv2
import numpy as np
from pylab import *
from matplotlib import pyplot as plt
import scipy.signal as signal
import detect_peaks

class ImagePartition(object):
    """Partition the image to single number

    Args:
        image: Input image with RGB


    Returns:


    """

    def __init__(self, image):
        """Init operate."""

        self.RGB = image
        self.balRGB = image
        self.resultRGB = image
        self.nbRGB = image
        self.b, self.g, self.r = cv2.split(self.RGB)
        self.gray = []
        self.grayline1 = []
        self.grayline2 = []

    def white_balance(self, per):
        """Revise white balance of the image."""

        self.b, self.g, self.r = cv2.split(self.RGB)
        xs = self.r.shape[0]
        ys = self.r.shape[1]
        th_po = xs * ys * per
        th_po = int(th_po)

        gray = cv2.cvtColor(self.RGB, cv2.COLOR_BGR2GRAY)
        graysort = np.sort(gray, axis=None)

        th_val = graysort[th_po]
        rmean = np.mean(self.r[np.where(gray > th_val)])
        bmean = np.mean(self.b[np.where(gray > th_val)])
        gmean = np.mean(self.g[np.where(gray > th_val)])

        kb = (rmean + gmean + bmean) / (3 * bmean)
        kr = (rmean + gmean + bmean) / (3 * rmean)
        kg = (rmean + gmean + bmean) / (3 * gmean)
        br = np.uint16(self.r) * kr
        bg = np.uint16(self.g) * kg
        bb = np.uint16(self.b) * kb
        balRGB = cv2.merge([bb, bg, br])
        balRGB[np.where(balRGB > 255)] = 255

        self.balRGB = np.uint8(balRGB)

    def reshape_image(self, cap_limit, lower_limit):
        """Shrink the image size for follow-up operation."""

        self.b, self.g, self.r = cv2.split(self.balRGB)
        fRGB = self.balRGB
        LAB = cv2.cvtColor(fRGB, cv2.COLOR_BGR2LAB)
        cv2.imshow('LAB', LAB)
        cv2.imshow('labL', LAB[:, :, 0])
        V = LAB[:, :, 0]
        cv2.imshow('L', V)
        V = cv2.equalizeHist(V)

        # 高亮像素计数圈数字位置法
        vhigh = V.copy()
        vhigh[np.where(V < 100)] = 0
        row_sum = np.sum(vhigh, axis=1)
        col_sum = np.sum(vhigh, axis=0)

        row_bd = np.where(row_sum > 2200)[0]
        col_bd = np.where(col_sum > 200)[0]

        cps = 2
        vlow = V.copy()
        vlow = np.int16(vlow)
        vlow[np.where(V < 1)] = -1
        vlow[np.where(vlow > 0)] = 0
        # print(vlow)
        col_sum_zeros = np.sum(vlow, axis=0)
        # print(col_sum_zeros)
        col_sum_zeros = col_sum_zeros * -1
        counts = np.bincount(col_sum_zeros)
        mean_size_zeros = np.argmax(counts)
        # print(mean_size_zeros)
        # print(col_sum_zeros)
        # print(counts+cps*2+1)
        col_bd_zeros = np.where(col_sum_zeros < mean_size_zeros + 1)[0]
        # print(col_bd_zeros)

        # col_bd_zeros_diff = np.diff(col_bd_zeros)

        col_bd = col_bd[np.where(col_bd > col_bd_zeros[0])]
        col_bd = col_bd[np.where(col_bd < col_bd_zeros[-1])]

        row_f = max(0, row_bd[0] - cps)
        row_l = min(row_bd[-1] + cps, fRGB.shape[0])
        col_f = max(0, col_bd[0] - cps)
        col_l = min(col_bd[-1] + cps, fRGB.shape[1])

        self.nbRGB = fRGB[row_f:row_l, col_f:col_l, :]
        self.resultRGB = self.RGB[row_f:row_l, col_f:col_l, :]

        cv2.imshow('nbRGB', self.nbRGB)

    def split_lines(self):
        """Split the image into line1 and line2."""

        LAB = cv2.cvtColor(self.nbRGB, cv2.COLOR_BGR2LAB)
        V = LAB[:, :, 0]
        cv2.imshow('L', V)
        # (h, w) = self.nbRGB.shape[:2]
        HistV = cv2.equalizeHist(V)
        LAB[:, :, 0] = HistV
        histRBG = cv2.cvtColor(LAB, cv2.COLOR_LAB2BGR)
        self.gray = cv2.cvtColor(histRBG, cv2.COLOR_BGR2GRAY)

        # 边缘检测部分
        gaussGray = cv2.GaussianBlur(gray, (5, 5), 5)
        # gaussGray = gray
        lapGray = cv2.Laplacian(gaussGray, cv2.CV_64F, ksize=3)
        lapGray = np.uint8(lapGray)
        # lapGray = np.uint8(np.absolute(lapGray))
        cv2.imshow('lapGray', lapGray)

        sobelx = cv2.Sobel(gaussGray, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.uint8(sobelx)
        cv2.imshow('sobelx', sobelx)

        sobely = cv2.Sobel(gaussGray, cv2.CV_64F, 0, 1, ksize=3)
        sobely = np.uint8(sobely)
        cv2.imshow('sobely', sobely)
        # histGray = cv2.cvtColor(HistRGB,cv2.COLOR_BGR2GRAY)
        histGray = self.gray
        lhGray = cv2.add(lapGray * 0.1, histGray * 0.9)
        lhGray = np.uint8(lhGray)
        cv2.imshow('lapGray', lapGray)
        cv2.imshow('histGray', histGray)
        cv2.imshow('lhGray', lhGray)

        row_sum2 = np.sum(lhGray, axis=1)
        # row_sum2 = np.sum(lapGray, axis=1)
        # row_sum2 = np.sum(gray,axis=1)
        # row_diff2 = np.diff(np.int64(row_sum2))
        x0 = self.gray.shape[0] // 2
        row_sump = row_sum2[x0 - 5:x0 + 6]
        row_bd2 = np.where(row_sum2 == np.min(row_sump))[0][0]  # 隐藏bug 万一有多个值

        l1h = row_bd2
        l2h = self.gray.shape[0] - l1h

        if l1h / l2h > 0.65 or l2h / l1h > 0.65:
            row_bd2 = self.gray.shape[0] // 2

        # 截取第一行和第二行
        rdt = 1
        self.grayline1 = self.gray[0:row_bd2 + rdt + 1, :]
        self.grayline2 = self.gray[row_bd2 - rdt:, :]

        self.resultRGB1 = self.resultRGB[0:row_bd2 + rdt + 1, :]
        self.resultRGB2 = self.resultRGB[row_bd2 - rdt:, :]

        cv2.imshow('grayline1', self.grayline1)
        cv2.imshow('grayline2', self.grayline2)

    @classmethod
    def part_hist(cls, grayline, peaks_line):
        fpx = 0
        HistV = np.zeros(grayline.shape)
        lp = len(peaks_line)
        for i, ppx in enumerate(peaks_line):
            if i == lp:
                ppx += 1
            HistV[:, fpx:ppx] = cv2.equalizeHist(grayline[:, fpx:ppx])
            fpx = ppx
        return np.uint8(HistV)

    def image_part_hist(self):
        pass




