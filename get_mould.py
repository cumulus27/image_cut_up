import numpy as np
import cv2
from pylab import *
from matplotlib import pyplot as plt
import scipy.signal as signal
from skimage import data, draw, color, transform, feature
import detect_peaks

# src="/home/py/PycharmProjects/image_process/extract/000003.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000008.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000048.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000006.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000012.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000824.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000027.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000032.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000019.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000118.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000055.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000075.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000153.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000164.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000158.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000177.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000184.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000200.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000205.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000212.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000214.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000216.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000229.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000230.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000246.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000259.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000278.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000286.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000356.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000724.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000063.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000874.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000877.jpg"
src="/home/py/PycharmProjects/image_process/extract/000924.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000067.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000150.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000003.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000018.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000041.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000097.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000918.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000930.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000926.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000907.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000898.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000887.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000859.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000851.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000872.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000845.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000843.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000834.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000830.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000825.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000821.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000655.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000611.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000585.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000569.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000548.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000513.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000509.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000501.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000488.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000483.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000463.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000447.jpg"


RGB = cv2.imread(src)
cv2.imshow('RGB', RGB)
(h, w) = RGB.shape[:2]
# RGB = cv2.GaussianBlur(RGB,(3,3),0.5)
# RGB = cv2.medianBlur(RGB,3)
# cv2.imshow('mediaRGB',RGB)

b, g, r = cv2.split(RGB)
'''
cv2.imshow('rgbR',r)   
cv2.imshow('rgbG',g)
cv2.imshow('rgbB',b)
'''

# white balance
per = 0.95
xs = r.shape[0]
ys = r.shape[1]
th_po = xs * ys * per
th_po = int(th_po)

gray = cv2.cvtColor(RGB, cv2.COLOR_BGR2GRAY)
graysort = np.sort(gray, axis=None)

th_val = graysort[th_po]
rmean = np.mean(r[np.where(gray > th_val)])
bmean = np.mean(b[np.where(gray > th_val)])
gmean = np.mean(g[np.where(gray > th_val)])

kb = (rmean + gmean + bmean) / (3 * bmean)
kr = (rmean + gmean + bmean) / (3 * rmean)
kg = (rmean + gmean + bmean) / (3 * gmean)

print(kb)
print(kr)
print(kg)


br = np.uint16(r) * kr
bg = np.uint16(g) * kg
bb = np.uint16(b) * kb
balRGB = cv2.merge([bb, bg, br])
balRGB[np.where(balRGB > 255)] = 255


balRGB = np.uint8(balRGB)
cv2.imshow('BalRGB ', balRGB)
# balRGB = cv2.medianBlur(balRGB, 3)
# cv2.imshow('mediaBalRGB ', balRGB)

b, g, r = cv2.split(balRGB)

# 直方图均衡化
fRGB = balRGB
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
col_bd_zeros = np.where(col_sum_zeros < mean_size_zeros+1)[0]
# print(col_bd_zeros)

# col_bd_zeros_diff = np.diff(col_bd_zeros)




col_bd = col_bd[np.where(col_bd > col_bd_zeros[0])]
col_bd = col_bd[np.where(col_bd < col_bd_zeros[-1])]

row_f = max(0, row_bd[0] - cps)
row_l = min(row_bd[-1] + cps, fRGB.shape[0])
col_f = max(0, col_bd[0] - cps)
col_l = min(col_bd[-1] + cps, fRGB.shape[1])

nbRGB = fRGB[row_f:row_l, col_f:col_l, :]
resultRGB = RGB[row_f:row_l, col_f:col_l, :]

cv2.imshow('nbRGB', nbRGB)
LAB = cv2.cvtColor(nbRGB, cv2.COLOR_BGR2LAB)
V = LAB[:, :, 0]
cv2.imshow('L', V)

(h, w) = nbRGB.shape[:2]

'''
# fen kuai jun heng
blockxnum = 16
blockynum = 2
blockx = r.shape[1] // blockxnum
blocky = r.shape[0] // blockynum

HistV = np.zeros(V.shape)
for i in range(blockxnum):
    for j in range(blockynum):
        fpx = i * blockx
        fpy = j * blocky
        if i == blockxnum - 1:
            ppx = V.shape[1]
        else:
            ppx = (i + 1) * blockx
        if j == blockynum - 1:
            ppy = V.shape[0]
        else:
            ppy = (j + 1) * blocky

        HistV[fpy:ppy, fpx:ppx] = cv2.equalizeHist(V[fpy:ppy, fpx:ppx])
HistV = np.uint8(HistV)
# HistRGB = cv2.GaussianBlur(HistRGB,(3,3),0.5)
cv2.imshow('HistV', HistV)
'''
HistV = cv2.equalizeHist(V)
LAB[:, :, 0] = HistV
histRBG = cv2.cvtColor(LAB, cv2.COLOR_LAB2BGR)
gray = cv2.cvtColor(histRBG, cv2.COLOR_BGR2GRAY)

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
histGray = gray
lhGray = cv2.add(lapGray * 0.1, histGray * 0.9)
lhGray = np.uint8(lhGray)
cv2.imshow('lapGray', lapGray)
cv2.imshow('histGray', histGray)
cv2.imshow('lhGray', lhGray)

row_sum2 = np.sum(lhGray, axis=1)
# row_sum2 = np.sum(lapGray, axis=1)
# row_sum2 = np.sum(gray,axis=1)
# row_diff2 = np.diff(np.int64(row_sum2))
x0 = gray.shape[0] // 2
row_sump = row_sum2[x0 - 5:x0 + 6]
row_bd2 = np.where(row_sum2 == np.min(row_sump))[0][0]  # 隐藏bug 万一有多个值

# 截取第一行和第二行
rdt = 1
grayline1 = gray[0:row_bd2 + rdt + 1, :]
grayline2 = gray[row_bd2 - rdt:, :]



#  保存模板
cv2.destroyAllWindows()
oooooo = grayline2
cv2.imshow('mould', oooooo)
mould = oooooo[3:22,118:133]
cv2.imshow('result', mould)
# mould = np.uint8(mould)
cv2.imwrite("/home/py/PycharmProjects/image_cut_up/mould/6/924_2_1.jpg", mould, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


cv2.waitKey(0)
cv2.destroyAllWindows()
