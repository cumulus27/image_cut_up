import numpy as np
import cv2
from pylab import *
from matplotlib import pyplot as plt
import scipy.signal as signal
from skimage import data, draw, color, transform, feature
import detect_peaks

# src="/home/py/PycharmProjects/image_process/extract/000003.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000048.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000006.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000012.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000824.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000027.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000019.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000118.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000003.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000055.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000075.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000158.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000177.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000200.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000205.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000212.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000216.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000229.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000246.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000259.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000724.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000063.jpg"
src = "/home/py/PycharmProjects/image_process/extract/000877.jpg"
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
per = 0.92
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
br = r * kr
bg = g * kg
bb = b * kb

balRGB = cv2.merge([bb, bg, br])
balRGB = np.uint8(balRGB)
cv2.imshow('BalRGB ', balRGB)
balRGB = cv2.medianBlur(balRGB, 3)
cv2.imshow('mediaBalRGB ', balRGB)

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

resultRGB1 = resultRGB[0:row_bd2 + rdt + 1, :]
resultRGB2 = resultRGB[row_bd2 - rdt:, :]

cv2.imshow('grayline1', grayline1)


'''
grayhigh1 = grayline1.copy()
grayhigh2 = grayline2.copy()

grayhigh1[np.where(grayline1 < 160)] = 0
grayhigh2[np.where(grayline2 < 160)] = 0

col_sh1 = np.sum(grayhigh1,axis=0)
col_sh2 = np.sum(grayhigh2,axis=0)

# plt.plot(col_sh1,'b')
# plt.plot(col_sh2,'r')
# plt.show()

col_bd1 = np.where(col_sh1 > 500)[0]
col_bd2 = np.where(col_sh2 > 500)[0]
# print(col_bd1)
# print(col_bd2)

# 尺寸确认
# print('尺寸确认0：')
# print(grayline1.shape)
# print(resultRGB1.shape)
# print(grayline2.shape)
# print(resultRGB2.shape)

cps = 3
# print(grayline1.shape)
gshape1 = grayline1.shape
gshape2 = grayline2.shape
grayline1 = grayline1[:,max(col_bd1[0]-cps,0):min(col_bd1[-1]+cps,gshape1[1])]
grayline2 = grayline2[:,max(col_bd2[0]-cps,0):min(col_bd2[-1]+cps,gshape2[1])]

cv2.imshow('grayline1_2',grayline1)
cv2.imshow('grayline2_2',grayline2)
resultRGB1 = resultRGB1[:,max(col_bd1[0]-cps,0):min(col_bd1[-1]+cps,gshape1[1]),:]
resultRGB2 = resultRGB2[:,max(col_bd2[0]-cps,0):min(col_bd2[-1]+cps,gshape2[1]),:]
# print(grayline1.shape)
'''
# 尺寸确认
# print('尺寸确认01：')
# print(grayline1.shape)
# print(resultRGB1.shape)
# print(grayline2.shape)
# print(resultRGB2.shape)


# 二次分块均衡
shd = 170
grayline1od = grayline1.copy()
grayline2od = grayline2.copy()
grayline1hd = grayline1.copy()
grayline2hd = grayline2.copy()
grayline1hd[np.where(grayline1 < shd)] //= 2
grayline2hd[np.where(grayline2 < shd)] //= 2
col_sum_line1 = np.sum(grayline1hd, axis=0)
col_sum_line2 = np.sum(grayline2hd, axis=0)

# 滤波
ksize = 3
col_sum_line1 = signal.medfilt(col_sum_line1, ksize)
col_sum_line2 = signal.medfilt(col_sum_line2, ksize)

col_sum_line1d = np.max(col_sum_line1) - col_sum_line1
col_sum_line2d = np.max(col_sum_line2) - col_sum_line2
# plt.plot(col_sum_line1d,'r')

peaks_line1 = signal.find_peaks_cwt(col_sum_line1d, np.arange(1, 14))
peaks_line2 = signal.find_peaks_cwt(col_sum_line2d, np.arange(1, 14))

# peaks_line1 = detect_peaks.detect_peaks(col_sum_line1d, mph=300, mpd=3, threshold=10)
# peaks_line2 = detect_peaks.detect_peaks(col_sum_line2d, mph=300, mpd=3, threshold=10)

peaks_line1 = np.concatenate((np.array([0]), peaks_line1, np.array([len(col_sum_line1d) - 1])), axis=0)
peaks_line2 = np.concatenate((np.array([0]), peaks_line2, np.array([len(col_sum_line2d) - 1])), axis=0)
print('第二次局部均衡分界点：')
print(peaks_line1)
print(peaks_line2)


def part_hist(grayline, peaks_line):
    fpx = 0
    HistV = np.zeros(grayline.shape)
    lp = len(peaks_line)
    for i, ppx in enumerate(peaks_line):
        if i == lp:
            ppx += 1
        HistV[:, fpx:ppx] = cv2.equalizeHist(grayline[:, fpx:ppx])
        fpx = ppx
    return np.uint8(HistV)


grayline1 = part_hist(grayline1, peaks_line1)
grayline2 = part_hist(grayline2, peaks_line2)

cv2.imshow('grayline1 part hist', grayline1)
cv2.imshow('grayline2 part hist', grayline2)

# 求和部分加入阈值
shd = 160
grayline1hd = grayline1.copy()
grayline2hd = grayline2.copy()
grayline1hd[np.where(grayline1 < shd)] //= 2
grayline2hd[np.where(grayline2 < shd)] //= 2
col_sum_line1 = np.sum(grayline1hd, axis=0)
col_sum_line2 = np.sum(grayline2hd, axis=0)

# 滤波
ksize = 3
col_sum_line1 = signal.medfilt(col_sum_line1, ksize)
col_sum_line2 = signal.medfilt(col_sum_line2, ksize)

col_sum_line1d = np.max(col_sum_line1) - col_sum_line1
col_sum_line2d = np.max(col_sum_line2) - col_sum_line2
# plt.plot(col_sum_line1d,'r')

# peaks_line1 = signal.find_peaks_cwt(col_sum_line1d, np.arange(1, 9))
# peaks_line2 = signal.find_peaks_cwt(col_sum_line2d, np.arange(1, 9))
#
peaks_line1 = detect_peaks.detect_peaks(col_sum_line1d, mph=300, mpd=5, threshold=0)
peaks_line2 = detect_peaks.detect_peaks(col_sum_line2d, mph=300, mpd=5, threshold=0)

peaks_line1 = np.concatenate((np.array([0]), peaks_line1, np.array([len(col_sum_line1d) - 1])), axis=0)
peaks_line2 = np.concatenate((np.array([0]), peaks_line2, np.array([len(col_sum_line2d) - 1])), axis=0)

print('尺寸确认：')
print(grayline1.shape)
print(col_sum_line1d.shape)
print(grayline2.shape)
print(col_sum_line2d.shape)
cv2.imshow('size confirm grayline1',grayline1)
cv2.imshow('size confirm grayline2',grayline2)
# peaks_high1 = col_sum_line1d[peaks_line1]
# peaks_high2 = col_sum_line1d[peaks_line2]

peaks_diff1 = np.diff(peaks_line1)
peaks_diff2 = np.diff(peaks_line2)


def get_bd_rawsum_diff(peaks_line, grayline):
    bd_diff = []
    for i, line in enumerate(peaks_line):
        pl = float64(grayline[:,line])
        pl_diff = np.diff(pl)
        pl_diff_abs = [i**2 for i in pl_diff]
        bd_diff.append(sum(pl_diff_abs))
    return bd_diff


bd_diff1 = get_bd_rawsum_diff(peaks_line1, grayline1)
bd_diff2 = get_bd_rawsum_diff(peaks_line2, grayline2)
print('bd_diff:')
print(bd_diff1)
print(bd_diff2)

def get_mser_line(grayline):
    mser = cv2.MSER_create(_min_area=40, _max_area=90)
    regions, boxes = mser.detectRegions(grayline)
    mser_line1 = []
    mser_line2 = []
    wrange = 3
    hrange = 3
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
    ww = []
    for box in boxes:
        x, y, w1, h1 = box
        if w1 < wmax[0] and w1 > wmin[0] and h1 > hmin[0] and h1 < hmax[0]:
            if x in xn and y in yn:
                pass
            else:
                xn.append(x)
                yn.append(y)
                w1 = wmid
                h1 = hmid
                ww.append(w1)
                mser_line1.append(x)
                mser_line2.append(x+w1)

    return mser_line1, mser_line2, ww



mser_line11, mser_line12, mser_diff1 = get_mser_line(grayline1)
mser_line21, mser_line22, mser_diff2 = get_mser_line(grayline2)
print("MSER :")
print(mser_line11)
print(mser_line12)
print(mser_line21)
print(mser_line22)
mean_size = (np.mean(mser_diff1) + np.mean(mser_diff2))/2+2.5


def merge_mser_line(mser_line1, mser_line2, mean_size):
    mser_line1.sort()
    mser_line2.sort()
    print('排序后：')
    print(mser_line1)
    print(mser_line2)
    delete = [0] * len(mser_line1)
    bias = 3
    for i in range(len(mser_line1)):
        if i > 0 and mser_line1[i] < mser_line2[i-1]:
            if mser_line1[i] - mser_line1[i-1] < bias or mser_line2[i] - mser_line2[i-1] < bias:
                mser_line1[i] = mser_line1[i - 1]
                mser_line1[i - 1] = -1
                mser_line2[i - 1] = -1
                # print(i)
            elif mser_line2[i-1] - mser_line1[i] > bias:
                print("warning!!! MSER结果出现交叉重叠！！")
                print(i)
                if i < len(mser_line1) - 1 and i-1 > 0:
                    af = abs(mser_line1[i-1] - mser_line2[i-2])
                    bf = abs(mser_line1[i] - mser_line2[i - 2])
                    ap = abs(mser_line1[i + 1] - mser_line2[i - 1])
                    bp = abs(mser_line1[i + 1] - mser_line2[i])
                    afh = abs(mser_line1[i-1] - mser_line1[i-2])
                    bfh = abs(mser_line1[i] - mser_line1[i - 2])
                    aph = abs(mser_line1[i + 1] - mser_line1[i - 1])
                    bph = abs(mser_line1[i + 1] - mser_line1[i])
                    if af <= bias or ap <= bias:
                        if bf > bias and bp > bias:
                            # 认为i-1是正确位置，除掉i
                            mser_line1[i] = mser_line1[i-1]
                            mser_line2[i] = mser_line2[i - 1]
                            delete[i] = 1
                        else:
                            print('warning!!! 两个位置都有非常接近的相邻块')
                            print(i)
                            if af < bias and bf < bias:
                                if afh > bfh:
                                    # 认为i-1是正确位置，除掉i
                                    mser_line1[i] = mser_line1[i - 1]
                                    mser_line2[i] = mser_line2[i - 1]
                                    delete[i] = 1
                                else:
                                    # 认为i是正确位置，除掉i-1
                                    mser_line1[i - 1] = mser_line1[i]
                                    mser_line2[i - 1] = mser_line2[i]
                            elif ap < bias and bp < bias:
                                if aph > bph:
                                    # 认为i-1是正确位置，除掉i
                                    mser_line1[i] = mser_line1[i - 1]
                                    mser_line2[i] = mser_line2[i - 1]
                                    delete[i] = 1
                                else:
                                    # 认为i是正确位置，除掉i-1
                                    mser_line1[i - 1] = mser_line1[i]
                                    mser_line2[i - 1] = mser_line2[i]
                            else:
                                print('出现设定外情景，请检查！！')

                    elif bf <= bias or bp <= bias:
                        if af > bias and ap > bias:
                            # 认为i是正确位置，除掉i-1
                            mser_line1[i-1] = mser_line1[i]
                            mser_line2[i-1] = mser_line2[i]
                        else:
                            print('warning!!! 两个位置都有非常接近的相邻块')
                            print(i)
                            if af < bias and bf < bias:
                                if afh > bfh:
                                    # 认为i-1是正确位置，除掉i
                                    mser_line1[i] = mser_line1[i - 1]
                                    mser_line2[i] = mser_line2[i - 1]
                                    delete[i] = 1
                                else:
                                    # 认为i是正确位置，除掉i-1
                                    mser_line1[i - 1] = mser_line1[i]
                                    mser_line2[i - 1] = mser_line2[i]
                            elif ap < bias and bp < bias:
                                if aph > bph:
                                    # 认为i-1是正确位置，除掉i
                                    mser_line1[i] = mser_line1[i - 1]
                                    mser_line2[i] = mser_line2[i - 1]
                                    delete[i] = 1
                                else:
                                    # 认为i是正确位置，除掉i-1
                                    mser_line1[i - 1] = mser_line1[i]
                                    mser_line2[i - 1] = mser_line2[i]
                            else:
                                print('出现设定外情景，请检查！！')
                    else:
                        # 两个位置都没有相邻的块来判断 用余数决胜负
                        afy = af % mean_size
                        bfy = bf % mean_size
                        apy = ap % mean_size
                        bpy = bp % mean_size
                        ay = min(afy, apy)
                        by = min(bfy, bpy)
                        if ay > by:
                            # 认为i是正确位置，除掉i-1
                            mser_line1[i - 1] = mser_line1[i]
                            mser_line2[i - 1] = mser_line2[i]
                            delete[i - 1] = 1
                        elif by > ay:
                            # 认为i-1是正确位置，除掉i
                            mser_line1[i] = mser_line1[i - 1]
                            mser_line2[i] = mser_line2[i - 1]
                            delete[i] = 1
                        else:
                            print('warning!!余数相同无法判断')
                            print(i)
                elif i-1 > 0:
                    af = abs(mser_line1[i - 1] - mser_line2[i - 2])
                    bf = abs(mser_line1[i] - mser_line2[i - 2])
                    afh = abs(mser_line1[i - 1] - mser_line1[i - 2])
                    bfh = abs(mser_line1[i] - mser_line1[i - 2])
                    if af <= bias:
                        if bf > bias:
                            # 认为i-1是正确位置，除掉i
                            mser_line1[i] = mser_line1[i - 1]
                            mser_line2[i] = mser_line2[i - 1]
                            delete[i] = 1
                        else:
                            print('warning!!! 两个位置都有非常接近的相邻块')
                            print(i)
                            if afh > bfh:
                                # 认为i-1是正确位置，除掉i
                                mser_line1[i] = mser_line1[i - 1]
                                mser_line2[i] = mser_line2[i - 1]
                                delete[i] = 1
                            else:
                                # 认为i是正确位置，除掉i-1
                                mser_line1[i - 1] = mser_line1[i]
                                mser_line2[i - 1] = mser_line2[i]
                    elif bf <= bias:
                        if af > bias:
                            # 认为i是正确位置，除掉i-1
                            mser_line1[i - 1] = mser_line1[i]
                            mser_line2[i - 1] = mser_line2[i]
                            delete[i - 1] = 1
                        else:
                            print('warning!!! 两个位置都有非常接近的相邻块')
                            print(i)
                            if afh > bfh:
                                # 认为i-1是正确位置，除掉i
                                mser_line1[i] = mser_line1[i - 1]
                                mser_line2[i] = mser_line2[i - 1]
                                delete[i] = 1
                            else:
                                # 认为i是正确位置，除掉i-1
                                mser_line1[i - 1] = mser_line1[i]
                                mser_line2[i - 1] = mser_line2[i]

                    else:
                        # 两个位置都没有相邻的块来判断 用余数决胜负
                        afy = af % mean_size
                        bfy = bf % mean_size
                        if afy > bfy:
                            # 认为i是正确位置，除掉i-1
                            mser_line1[i - 1] = mser_line1[i]
                            mser_line2[i - 1] = mser_line2[i]
                            delete[i-1] = 1
                        elif bfy > afy:
                            # 认为i-1是正确位置，除掉i
                            mser_line1[i] = mser_line1[i - 1]
                            mser_line2[i] = mser_line2[i - 1]
                            delete[i] = 1
                        else:
                            print('warning!!余数相同无法判断')
                            print(i)
                elif i < len(mser_line1) - 1:
                    ap = abs(mser_line1[i + 1] - mser_line2[i - 1])
                    bp = abs(mser_line1[i + 1] - mser_line2[i])
                    aph = abs(mser_line1[i + 1] - mser_line1[i - 1])
                    bph = abs(mser_line1[i + 1] - mser_line1[i])
                    if ap <= bias:
                        if bp > bias:
                            # 认为i-1是正确位置，除掉i
                            mser_line1[i] = mser_line1[i - 1]
                            mser_line2[i] = mser_line2[i - 1]
                            delete[i] = 1
                        else:
                            print('warning!!! 两个位置都有非常接近的相邻块')
                            print(i)
                            if aph > bph:
                                # 认为i-1是正确位置，除掉i
                                mser_line1[i] = mser_line1[i - 1]
                                mser_line2[i] = mser_line2[i - 1]
                                delete[i] = 1
                            else:
                                # 认为i是正确位置，除掉i-1
                                mser_line1[i - 1] = mser_line1[i]
                                mser_line2[i - 1] = mser_line2[i]
                    elif bp <= bias:
                        if ap > bias:
                            # 认为i是正确位置，除掉i-1
                            mser_line1[i - 1] = mser_line1[i]
                            mser_line2[i - 1] = mser_line2[i]
                            delete[i - 1] = 1
                        else:
                            print('warning!!! 两个位置都有非常接近的相邻块')
                            print(i)
                            if aph > bph:
                                # 认为i-1是正确位置，除掉i
                                mser_line1[i] = mser_line1[i - 1]
                                mser_line2[i] = mser_line2[i - 1]
                                delete[i] = 1
                            else:
                                # 认为i是正确位置，除掉i-1
                                mser_line1[i - 1] = mser_line1[i]
                                mser_line2[i - 1] = mser_line2[i]
                    else:
                        # 两个位置都没有相邻的块来判断 用余数决胜负
                        apy = ap % mean_size
                        bpy = bp % mean_size
                        if apy > bpy:
                            # 认为i是正确位置，除掉i-1
                            mser_line1[i - 1] = mser_line1[i]
                            mser_line2[i - 1] = mser_line2[i]
                            delete[i-1] = 1
                        elif bpy > apy:
                            # 认为i-1是正确位置，除掉i
                            mser_line1[i] = mser_line1[i - 1]
                            mser_line2[i] = mser_line2[i - 1]
                            delete[i] = 1
                        else:
                            print('warning!!余数相同无法判断')
                            print(i)
                else:
                    print('warning!!!未知情景，请检查！')

    print(mser_line1)
    print(mser_line2)
    print(delete)
    mser_line1 = [i for i,j in zip(mser_line1, delete) if i != -1 and j != 1]
    mser_line2 = [i for i,j in zip(mser_line2, delete) if i != -1 and j != 1]
    print(mser_line1)
    print(mser_line2)
    return mser_line1, mser_line2

mser_line11, mser_line12 = merge_mser_line(mser_line11, mser_line12, mean_size)
mser_line21, mser_line22 = merge_mser_line(mser_line21, mser_line22, mean_size)
print("MSER 排序去重 :")
print(mser_line11)
print(mser_line12)
print(mser_line21)
print(mser_line22)


'''
mser = cv2.MSER_create(_min_area=40, _max_area=90)
regions, boxes = mser.detectRegions(grayline1)
temp1 = np.zeros(grayline1.shape)
wrange = 4
hrange = 4
histnum_w = hist(boxes[:, 2])
wmax_index = find(histnum_w[0][:] == max(histnum_w[0][:]))
wmin = histnum_w[1][wmax_index] - wrange
wmax = histnum_w[1][wmax_index] + wrange
histnum_h = hist(boxes[:, 3])
hmax_index = find(histnum_h[0][:] == max(histnum_h[0][:]))
hmin = histnum_h[1][hmax_index] - hrange
hmax = histnum_h[1][hmax_index] + hrange
wmid=int((wmax[0]+wmin[0])/2)+2
hmid=int((hmax[0]+hmin[0])/2)+3
xn=[]
yn=[]
for box in boxes:
    x, y, w1, h1 = box
    if w1 < wmax[0] and w1 > wmin[0] and h1 > hmin[0] and h1 < hmax[0]:
        if x in xn and y in yn:
            pass
        else:
            xn.append(x)
            yn.append(y)
            w1=wmid
            h1=hmid
            temp1[y:y + h1, x:x + w1] = 1
temp1=temp1*grayline1
temp1=np.uint8(temp1)
cv2.imshow('temp1',temp1)
# print('mser:')
# print(regions)
# print(boxes)
'''

def plot_bd_rawsum(peaks_line, grayline):
    for i, line in enumerate(peaks_line):
        pl = float64(grayline[:,line])
        pl = np.add(np.ones(pl.shape)*255*i, pl)
        plt.plot(pl, 'r')
    plt.show()


# plot_bd_rawsum(peaks_line1, grayline1)
# plot_bd_rawsum(peaks_line2, grayline2)
#
# 尺寸确认
# print('尺寸确认：')
# print(grayline1.shape)
# print(resultRGB1.shape)
# print(grayline2.shape)
# print(resultRGB2.shape)
# 中位数做参考值
diff_queue = np.append(peaks_diff1, peaks_diff2)
diff_queue = diff_queue[np.where(diff_queue > 8)]
diff_queue = diff_queue[np.where(diff_queue < 16)]

# mean_size = np.mean(diff_queue)
# mean_size = np.median(diff_queue)
mean_size = (np.mean(mser_diff1) + np.mean(mser_diff2))/2+2.5


# 众数做参考值
# counts = np.bincount(diff_queue)
# mean_size = np.argmax(counts)

trust1 = np.ones(peaks_line1.shape)*4
trust2 = np.ones(peaks_line2.shape)*4


# 模板匹配
def mould_detect(img2, template, methods):
    ww, hh = template.shape[::-1]
    img = img2.copy()
    method = eval(methods)
    res = cv2.matchTemplate(img, template, method)
    # print('模板匹配结果：')
    # print(res)
    loc_f = []
    loc_p = []

    i = 0
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        threshold = 0.1
        loc = np.where(res <= threshold)
        for pt in zip(*loc[::-1]):
            loc_f.append(pt[0])
            loc_p.append(pt[0] + ww)
            i += 1
            cv2.rectangle(img, pt, (pt[0] + ww, pt[1] + hh), 255, 2)
            # while
    else:
        threshold = 0.9
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            loc_f.append(pt[0])
            loc_p.append(pt[0] + ww)
            i += 1
            cv2.rectangle(img, pt, (pt[0] + ww, pt[1] + hh), 255, 2)

    print(methods)
    print('个数：')
    print(i)

    plt.subplot(221), plt.imshow(img2, cmap="gray")
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(template, cmap="gray")
    plt.title('template Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(res, cmap="gray")
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(img, cmap="gray")
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.show()

    # 重复过滤
    loc_f2 = []
    loc_p2 = []
    bias = 4
    for f,p in zip(loc_f, loc_p):
        flag = 0
        for i,j in zip(loc_f2, loc_p2):
            if abs(f-i) < bias and abs(p-j) < bias:
                flag = 1
            elif abs(f-i) < bias or abs(p-j) < bias:
                print('err!!! 模板匹配结果存在错误！ ')

        if flag == 0:
            loc_f2.append(f)
            loc_p2.append(p)

    return loc_f2, loc_p2



def detect_number(img2):
    # tempNum = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    tempNum = ['0', '1']
    line_number_f = []
    line_number_p = []
    for num in tempNum:
        line_number_f0 = []
        line_number_p0 = []
        template1 = cv2.imread("/home/py/PycharmProjects/image_cut_up/mould/{}/001.jpg".format(num), 0)
        template2 = cv2.imread("/home/py/PycharmProjects/image_cut_up/mould/{}/002.jpg".format(num), 0)
        template3 = cv2.imread("/home/py/PycharmProjects/image_cut_up/mould/{}/003.jpg".format(num), 0)
        template_list = ['template1', 'template2', 'template3']
        methods = 'cv2.TM_SQDIFF_NORMED'
        for temp in template_list:
            template = eval(temp)
            lf, lp = mould_detect(img2, template, methods)
            line_number_f0.append(lf)
            line_number_p0.append(lp)

        line_number_f.append(line_number_f0)
        line_number_p.append(line_number_p0)

    return line_number_f, line_number_p


line_number_f1, line_number_p1 =  detect_number(grayline1od)
line_number_f2, line_number_p2 =  detect_number(grayline2od)


def mould_result_filter(line_number_f, line_number_p):
    line_number_sf = []
    line_number_sp = []
    maxnum = 6

    for singlef, singlep in zip(line_number_f, line_number_p):
        choice = 0
        current = 0
        print(singlef)
        for i,point in enumerate(singlef):
            if len(point) > current and len(point) <= maxnum:
                choice = i

        print('choice:')
        print(choice)

        line_number_sf.append(sort(singlef[choice]))
        line_number_sp.append(sort(singlep[choice]))

    return line_number_sf, line_number_sp


line_number_sf1, line_number_sp1 = mould_result_filter(line_number_f1, line_number_p1)
line_number_sf2, line_number_sp2 = mould_result_filter(line_number_f2, line_number_p2)

print('数字模板匹配结果：')
print(line_number_f1)
print(line_number_p1)
print(line_number_f2)
print(line_number_p2)


'''
img2 = grayline2od
# template1 = cv2.imread("/home/py/PycharmProjects/mould0/012_1_2.jpg", 0)
# template2 = cv2.imread("/home/py/PycharmProjects/mould0/877_2_1.jpg", 0)
# template3 = cv2.imread("/home/py/PycharmProjects/mould0/003_2_1.jpg", 0)
# template4 = cv2.imread("/home/py/PycharmProjects/mould0/003_2_6_1.jpg", 0)
# template5 = cv2.imread("/home/py/PycharmProjects/mould0/012_2_3_1.jpg", 0)
# template6 = cv2.imread("/home/py/PycharmProjects/mould0/877_1_1_1.jpg", 0)
# template_list = ['template1', 'template2', 'template3', 'template4', 'template5', 'template6']

# methods = ['cv2.TM_CCOEFF_NORMED',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

methods = 'cv2.TM_SQDIFF_NORMED'

for temp in template_list:
    template = eval(temp)
    print(template.shape)
    ww, hh = template.shape[::-1]

    img = img2.copy()
    method = eval(methods)
    res = cv2.matchTemplate(img, template, method)
    print('模板匹配结果：')
    print(res)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    i = 0

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        threshold = 0.1
        loc = np.where(res <= threshold)
        for pt in zip(*loc[::-1]):
            i += 1
            cv2.rectangle(img, pt, (pt[0] + ww, pt[1] + hh), 255, 2)
        # while
    else:
        threshold = 0.9
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            i += 1
            cv2.rectangle(img, pt, (pt[0] + ww, pt[1] + hh), 255, 2)
    # bottom_right = (top_left[0] + ww, top_left[1] + hh)
    # cv2.rectangle(img, top_left, bottom_right, 255, 2)

    print(methods)
    print('个数：')
    print(i)
    plt.subplot(221), plt.imshow(img2, cmap="gray")
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(template, cmap="gray")
    plt.title('template Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(res, cmap="gray")
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(img, cmap="gray")
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.show()
'''




num = 16
# mean_size = round(gray.shape[1]/num)
print('参考宽度：')
print(mean_size)
bias = round(mean_size * 0.25)


def get_diff_status(peaks_diff, mean_size, bias):
    sl = mean_size - bias
    ll = mean_size + bias
    n = len(peaks_diff)
    diff_status = np.zeros(peaks_diff.shape)
    for i in range(n):
        if peaks_diff[i] > ll:
            diff_status[i] = 2
        elif peaks_diff[i] < sl:
            diff_status[i] = 1
        elif peaks_diff[i] == 0:
            diff_status[i] = -1

    return diff_status


diff_status1 = get_diff_status(peaks_diff1, mean_size, bias)
diff_status2 = get_diff_status(peaks_diff2, mean_size, bias)

print('peaks line:')
print(peaks_line1)
print(peaks_line2)
print('First Peaks Diff:')
print(peaks_diff1)
print(peaks_diff2)
print('Init status:')
print(diff_status1)
print(diff_status2)
# plot_bd_rawsum(peaks_line1, grayline1)
# plot_bd_rawsum(peaks_line2, grayline2)


# 从分界点周围的宽度确定第一次置信度
def get_neighber_trust(trust, peaks_diff, diff_status):
    neighbers = np.zeros(trust.shape)
    n = len(peaks_diff)
    for i in range(n):
        if diff_status[i] == 0:
            neighbers[i] += 1
            neighbers[i + 1] += 1

    trust = neighbers * 4 + trust
    return trust


trust1 = get_neighber_trust(trust1, peaks_diff1, diff_status1)
trust2 = get_neighber_trust(trust2, peaks_diff2, diff_status2)

print('First Trust:')
print(trust1)
print(trust2)


# 将两种方案得到的分界线融合起来
def combine_the_two_way(peaks_line, mser_line1, mser_line2, trust):
    i = 0
    mser_line1.append(1000)
    for j,line in enumerate(peaks_line):
        while i < len(mser_line1)-1 and line > mser_line1[i+1]:
            i += 1

        if abs(mser_line1[i] - line) < bias or abs(mser_line2[i] - line) < bias:
            trust[j] += 8
        elif line - mser_line1[i] > bias + 1 and mser_line2[i] - line > bias + 1:
            trust[j] -= 9

    return trust


trust1 = combine_the_two_way(peaks_line1, mser_line11, mser_line12, trust1)
trust2 = combine_the_two_way(peaks_line2, mser_line21, mser_line22, trust2)
print('融合后的置信度：')
print(trust1)
print(trust2)

#从分界线本身像素分布确定第二次置信度
def get_self_trust(trust, grayline, peaks_line, thresholdh, thresholdl):
    graylinehd = grayline.copy()
    graylineld = grayline.copy()

    graylinehd[np.where(graylinehd < thresholdh)] = 0
    graylineld[np.where(graylineld < thresholdl)] = 0
    high = grayline.shape[0] // 3
    l1 = high
    l2 = high * 2
    part1 = sum(graylinehd[:l1,peaks_line], axis=0)
    part2 = sum(graylineld[l1:l2, peaks_line], axis=0)
    part3 = sum(graylinehd[l2:, peaks_line], axis=0)
    for i in range(len(trust)):
        if part1[i] > 200 and part3[i] > 200:
            if part2[i] < 200:
                if trust[i] < 8:
                    trust[i] -= 8
                else:
                    trust[i] -= 5
                # print('Get one zero:')
                # print(i)
                # print(peaks_line[i])
                # print(part1[i])
                # print(part2[i])
                # print(part3[i])
                # print(grayline[:,peaks_line[i]])
            else:
                trust[i] = trust[i] // 2

    return trust


thresholdh = 180
thresholdl = 170
trust1 = get_self_trust(trust1, grayline1, peaks_line1, thresholdh, thresholdl)
trust2 = get_self_trust(trust2, grayline2, peaks_line2, thresholdh, thresholdl)
print('Second Trust:')
print(trust1)
print(trust2)

'''
def delete_zeros(line_result, trust):

    for i, status in enumerate(trust):
        if i > 0 and i < len(trust)-1 and status < 0:
            mix = line_result[i - 1] + line_result[i]
            line_result[i - 1] = 0
            line_result[i] = mix

    return line_result


# peaks_diff1 = delete_zeros(peaks_diff1, trust1)
# peaks_diff2 = delete_zeros(peaks_diff2, trust2)
#
# trust1 = trust1[np.where(trust1 >= 0)]
# trust2 = trust2[np.where(trust2 >= 0)]
# peaks_diff1 = peaks_diff1[np.where(peaks_diff1 > 0)]
# peaks_diff2  = peaks_diff2[np.where(peaks_diff2 > 0)]

diff_status1 = get_diff_status(peaks_diff1, mean_size, bias)
diff_status2 = get_diff_status(peaks_diff2, mean_size, bias)

print('Delete Zeros:')
print(peaks_diff1)
print(peaks_diff2)
print(trust1)
print(trust2)
print(diff_status1)
print(diff_status2)
'''


def get_distrubute_trust(peaks_line, trust, peaks_diff, bd_diff, diff_status, mean_size):

    for i,line in enumerate(peaks_line):
        if i > 0 and i < len(peaks_line)-1:
            if diff_status[i-1] == 1 and diff_status[i] == 1:
                # 两边的分块都偏小 降低置信度
                trust[i] -= 2
                if i > 1 and diff_status[i-2] == 1:
                    if peaks_line[i+1] - peaks_line[i-2] > mean_size + 3:
                        if bd_diff[i-1] > bd_diff[i]:
                            trust[i-1] -= 4
                        else:
                            trust[i] -= 4
                    else:
                        trust[i - 1] -= 4
                        trust[i] -= 4
            elif diff_status[i-1] == 1 and diff_status[i] == 0:
                trust[i] -= 1
            elif diff_status[i - 1] == 0 and diff_status[i] == 1:
                trust[i] -= 1
            elif diff_status[i - 1] == 2 and diff_status[i] == 0:
                trust[i] -= 1
            elif diff_status[i - 1] == 0 and diff_status[i] == 2:
                trust[i] -= 1
            elif diff_status[i - 1] == 2 and diff_status[i] == 2:
                trust[i] -= 2
            elif diff_status[i - 1] == 0 and diff_status[i] == 0:
                trust[i] += 1

    return trust


trust1 = get_distrubute_trust(peaks_line1, trust1, peaks_diff1, bd_diff1, diff_status1, mean_size)
trust2 = get_distrubute_trust(peaks_line2, trust2, peaks_diff2, bd_diff2, diff_status2, mean_size)

print('get_distrubute_trust:')
print(peaks_diff1)
print(peaks_diff2)
print(trust1)
print(trust2)
print(diff_status1)
print(diff_status2)

# 从模板匹配的结果对 置信度进行最后的筛选
def merge_mould_result(peaks_line, trust, line_number_f, line_number_p, mean_size):
    for singlef, singlep in zip(line_number_f, line_number_p):
        i = 0
        print(singlef)
        if len(singlef) == 0:
            continue
        for j, line in enumerate(peaks_line):
            while i < len(singlef) - 1 and line > singlef[i + 1]:
                i += 1

            if abs(singlef[i] - line) < bias or abs(singlep[i] - line) < bias:
                trust[j] += 18
            elif line - singlef[i] > bias and singlep[i] - line > bias:
                trust[j] -= 19
            elif singlef[i] - line > bias+1 and singlef[i] - line < mean_size - 4:
                trust[j] -= 8
            elif line - singlef[i] > bias+1 and line - singlep[i] < mean_size - 4:
                trust[j] -= 8

    return trust



print(line_number_sf1)
trust1 = merge_mould_result(peaks_line1, trust1, line_number_sf1, line_number_sp1, mean_size)
trust2 = merge_mould_result(peaks_line2, trust2, line_number_sf2, line_number_sp2, mean_size)
print('融合模板匹配的结果后：')
print(trust1)
print(trust2)

peaks_line1 = peaks_line1[np.where(trust1 >= 0)]
peaks_line2 = peaks_line2[np.where(trust2 >= 0)]
trust1 = trust1[np.where(trust1 >= 0)]
trust2 = trust2[np.where(trust2 >= 0)]
peaks_diff1 = np.diff(peaks_line1)
peaks_diff2 = np.diff(peaks_line2)
diff_status1 = get_diff_status(peaks_diff1, mean_size, bias)
diff_status2 = get_diff_status(peaks_diff2, mean_size, bias)


print('删除掉被排除掉的点之后:')
print(peaks_line1)
print(peaks_line2)
print(trust1)
print(trust2)
print(peaks_diff1)
print(peaks_diff2)
print(diff_status1)
print(diff_status2)





def handle_small_diff(trust, peaks_diff, diff_status, mean_size, bias):
    n = len(peaks_diff)
    for i in range(n):
        if diff_status[i] == 1:
            if i < n - 1 and diff_status[i + 1] == 1:
                if peaks_diff[i + 1] + peaks_diff[i] < mean_size + bias + 2:
                    peaks_diff[i + 1] = peaks_diff[i + 1] + peaks_diff[i]
                    peaks_diff[i] = 0
                    diff_status = get_diff_status(peaks_diff, mean_size, bias)
                    trust[i] = -1
            elif i >= n - 1:
                if diff_status[i] < bias + 1:
                    peaks_diff[i] = peaks_diff[i - 1] + peaks_diff[i]
                    peaks_diff[i - 1] = 0
                    diff_status = get_diff_status(peaks_diff, mean_size, bias)
                    trust[i - 1] = -2
                    # peaks_diff[i] = 0
                    # trust[i] = -1
            elif i == 0 and peaks_diff[i + 1] + peaks_diff[i] < mean_size + bias:
                peaks_diff[i + 1] = peaks_diff[i + 1] + peaks_diff[i]
                peaks_diff[i] = 0
                diff_status = get_diff_status(peaks_diff, mean_size, bias)
                trust[i] = -3
            elif i == 0 and peaks_diff[i] < mean_size // 2:
                # 第一个分块偏小 且大小小于偏差值
                if trust[i+1] > 0:
                    pass
                else:
                    peaks_diff[i + 1] = peaks_diff[i + 1] + peaks_diff[i]
                    peaks_diff[i] = 0
                    diff_status = get_diff_status(peaks_diff, mean_size, bias)
                    trust[i] = -4
            elif i > 0 and diff_status[i - 1] == 0:
                # print(diff_status[i])
                # print(diff_status[i+1])
                if diff_status[i + 1] == 2:
                    peaks_diff[i + 1] = peaks_diff[i + 1] + peaks_diff[i]
                    peaks_diff[i] = 0
                    diff_status = get_diff_status(peaks_diff, mean_size, bias)
                    trust[i] = -5
                elif diff_status[i + 1] == 0:
                    if peaks_diff[i] < bias + 2:
                        # 超小块噪声 直接归并到旁边
                        if peaks_diff[i - 1] > peaks_diff[i + 1]:
                            peaks_diff[i + 1] = peaks_diff[i + 1] + peaks_diff[i]
                            peaks_diff[i] = 0
                            diff_status = get_diff_status(peaks_diff, mean_size, bias)
                            trust[i] = -6
                        else:
                            peaks_diff[i] = peaks_diff[i - 1] + peaks_diff[i]
                            peaks_diff[i - 1] = 0
                            diff_status = get_diff_status(peaks_diff, mean_size, bias)
                            trust[i - 1] = -7
                    else:
                        trust[i] = -8
                        print('err!!!本来预定的数字区中间夹了个非数字的小块')
                        print(i)
            elif diff_status[i + 1] == 0:
                if i > 0 and diff_status[i - 1] == 2:
                    peaks_diff[i] = peaks_diff[i - 1] + peaks_diff[i]
                    peaks_diff[i - 1] = 0
                    diff_status = get_diff_status(peaks_diff, mean_size, bias)
                    trust[i - 1] = -9
            elif i > 0 and diff_status[i + 1] == 2 and diff_status[i - 1] != 2:
                # 后面的位置偏大 前面的位置不偏大 把本位置和后面的位置合并
                peaks_diff[i + 1] = peaks_diff[i + 1] + peaks_diff[i]
                peaks_diff[i] = 0
                diff_status = get_diff_status(peaks_diff, mean_size, bias)
                trust[i] = -12
            elif i > 0 and diff_status[i + 1] == 2 and diff_status[i - 1] == 2:
                if trust[i] > trust[i+1]:
                    peaks_diff[i + 1] = peaks_diff[i + 1] + peaks_diff[i]
                    peaks_diff[i] = 0
                    diff_status = get_diff_status(peaks_diff, mean_size, bias)
                else:
                    peaks_diff[i] = peaks_diff[i - 1] + peaks_diff[i]
                    peaks_diff[i - 1] = 0
                    diff_status = get_diff_status(peaks_diff, mean_size, bias)
                print('warning!!!两边都是大区，使用置信度判断')
                print(i)
            else:
                trust[i] = -11
                print('err!!!未知错误，请检查！')
                print(i)

    return trust, peaks_diff, diff_status


trust1, peaks_diff1, diff_status1 = handle_small_diff(trust1, peaks_diff1, diff_status1, mean_size, bias)
trust2, peaks_diff2, diff_status2 = handle_small_diff(trust2, peaks_diff2, diff_status2, mean_size, bias)

# 第一轮之后的trust
print('第一轮之后的trust:')
print(trust1)
print(trust2)
line1_result = peaks_diff1[np.where(peaks_diff1 > 0)]
line2_result = peaks_diff2[np.where(peaks_diff2 > 0)]

'''

def handle_big_diff(trust, peaks_diff, diff_status, mean_size, bias):
    sl = mean_size - bias - 1
    ll = mean_size + bias
    i = 0
    for diff, status in zip(peaks_diff, diff_status):
        # print(i)
        # print(status)
        # print(diff)
        # print(peaks_diff[i])
        if status == 2:
            j = 1
            while True:
                j += 1
                fenjie = round(diff / j)
                # print(fenjie)
                if fenjie < ll + bias * 2 and fenjie > sl + 1:
                    # print(peaks_diff)
                    peaks_diff[i] = fenjie
                    for n in range(j - 2):
                        peaks_diff = np.insert(peaks_diff, i, fenjie)
                        i += 1
                    # print(peaks_diff)
                    peaks_diff = np.insert(peaks_diff, i, diff - (j - 1) * fenjie)
                    i += 1
                    # print(peaks_diff)
                    break
                elif fenjie > ll:
                    continue
                else:
                    break
        i += 1
    return peaks_diff


peaks_diff1 = handle_big_diff(trust1, peaks_diff1, diff_status1, mean_size, bias)
peaks_diff2 = handle_big_diff(trust2, peaks_diff2, diff_status2, mean_size, bias)

line1_result = peaks_diff1[np.where(peaks_diff1 > 0)]
line2_result = peaks_diff2[np.where(peaks_diff2 > 0)]

# print('result:')
# print(line1_result)
# print(line2_result)


# 两个连续偏大情况纠错
def three2two_err_fix(line_result, num, mean_size, bias):
    length = len(line_result)
    too_big = np.zeros(line_result.shape)
    too_big = np.append(too_big, too_big)
    if length >= 20:
        return line_result
    elif length < 20:
        i = 0
        for result in line_result:
            if result > mean_size + 2:
                # print(i)
                too_big[i] = 1
                # if i == 0:             # 首项不处理  暂时处理方法  前面搞定后去掉
                #     too_big[i] = 0
                if i > 0 and too_big[i - 1] == 1:
                    mix = result + line_result[i - 1]
                    sl = mean_size - bias
                    ll = mean_size + bias
                    trsplit = round(mix / 3)
                    if trsplit > sl - 2 and trsplit < ll:
                        line_result[i - 1] = trsplit
                        line_result[i] = trsplit
                        line_result = np.insert(line_result, i, mix - 2 * trsplit)
                        i += 1
                        too_big[i] = 2
                    elif trsplit > ll:
                        print('err! 分成三个还是太大，请检查')
            i += 1
    return line_result


# line1_result = three2two_err_fix(line1_result, num, mean_size, bias)
# line2_result = three2two_err_fix(line2_result, num, mean_size, bias)
#
# print('修正3to2bug后：')
# print(line1_result)
# print(line2_result)
#
# 连续两个一个偏大一个偏小情况纠正
diff_status1 = get_diff_status(line1_result, mean_size, bias)
diff_status2 = get_diff_status(line2_result, mean_size, bias)
print('result status:')
print(diff_status1)
print(diff_status2)


def big_small_fix(line_result, diff_status):
    for i, status in enumerate(diff_status):
        if i > 0 and ((status == 1 and diff_status[i - 1] == 2) or (status == 2 and diff_status[i - 1] == 1)):
            if i < diff_status.shape[0] - 1 and diff_status[i + 1] != 0:
                print('warning!!! 连续三个位置异常，处理可能出错')

            mix = line_result[i - 1] + line_result[i]
            line_result[i - 1] = mix // 2
            line_result[i] = mix - line_result[i - 1]
            diff_status[i] = -1

    return line_result


line1_result = big_small_fix(line1_result, diff_status1)
line2_result = big_small_fix(line2_result, diff_status2)

print('修正big_small bug后：')
print(line1_result)
print(line2_result)

# 相隔一个位置有一个小块的bug修复
diff_status1 = get_diff_status(line1_result, mean_size, bias)
diff_status2 = get_diff_status(line2_result, mean_size, bias)
print('result status:')
print(diff_status1)
print(diff_status2)


def small_big_small_fix(line_result, diff_status):
    for i, status in enumerate(diff_status):
        if i > 1 and (status == 1 and diff_status[i - 1] != 1 and diff_status[i - 2] == 1):
            if i < diff_status.shape[0] - 1 and diff_status[i + 1] != 0:
                print('warning!!! 后面位置仍然异常，处理可能出错')
            mix = line_result[i - 1] + line_result[i] + line_result[i - 2]
            line_result[i - 1] = mix // 2
            line_result[i - 2] = 0
            line_result[i] = mix - line_result[i - 1]
            diff_status[i] = -1

    return line_result


line1_result = small_big_small_fix(line1_result, diff_status1)
line2_result = small_big_small_fix(line2_result, diff_status2)

line1_result = line1_result[np.where(line1_result > 0)]
line2_result = line2_result[np.where(line2_result > 0)]
print('修正small_big_small bug后：')
print(line1_result)
print(line2_result)

# 连续三个小于平均数
# diff_status1 = get_diff_status(line1_result, mean_size, 0)
# diff_status2 = get_diff_status(line2_result, mean_size, 0)
# print('result status:')
# print(diff_status1)
# print(diff_status2)


def small_small_small_fix(line_result, diff_status):
    for i, status in enumerate(diff_status):
        if i > 1 and (status == 1 and diff_status[i - 1] == 1 and diff_status[i - 2] == 1):
            if i < diff_status.shape[0] - 1 and diff_status[i + 1] != 0:
                print('warning!!! 后面位置仍然异常，处理可能出错')
            mix = line_result[i - 1] + line_result[i] + line_result[i - 2]
            line_result[i - 1] = mix // 2
            line_result[i - 2] = 0
            line_result[i] = mix - line_result[i - 1]
            diff_status[i] = -1

    return line_result


# line1_result = small_small_small_fix(line1_result, diff_status1)
# line2_result = small_small_small_fix(line2_result, diff_status2)

# line1_result = line1_result[np.where(line1_result>0)]
# line2_result = line2_result[np.where(line2_result>0)]
# print('修正small_small_small bug后：')
# print(line1_result)
# print(line2_result)

# 连续三个小于平均数
# diff_status1 = get_diff_status(line1_result, mean_size, 0)
# diff_status2 = get_diff_status(line2_result, mean_size, 0)
# print('result status:')
# print(diff_status1)
# print(diff_status2)


def small_small_fix(line_result, diff_status):
    for i, status in enumerate(diff_status):
        if i > 1 and (status == 1 and diff_status[i - 1] == 1 and diff_status[i - 2] == 1):
            if i < diff_status.shape[0] - 1 and diff_status[i + 1] != 0:
                print('warning!!! 后面位置仍然异常，处理可能出错')
            mix = line_result[i - 1] + line_result[i] + line_result[i - 2]
            line_result[i - 1] = mix // 2
            line_result[i - 2] = 0
            line_result[i] = mix - line_result[i - 1]
            diff_status[i] = -1

    return line_result


# line1_result = small_small_fix(line1_result, diff_status1)
# line2_result = small_small_fix(line2_result, diff_status2)

# line1_result = line1_result[np.where(line1_result>0)]
# line2_result = line2_result[np.where(line2_result>0)]
# print('修正small_small bug后：')
# print(line1_result)
# print(line2_result)
'''

def show_result(first, line_result, resultRGB, line):
    point = first
    name = 1
    fat = 2
    for i in line_result:
        sigleim = resultRGB[:, max(point - fat, 0):min(point + i + fat, resultRGB.shape[1] - 1)]
        # print(point-fat)
        # print(point+i+fat)
        # print(resultRGB.shape)
        # print(sigleim.shape)
        cv2.imshow(str(line) + 'sigle' + str(name), sigleim)
        point += i
        name += 1


show_result(peaks_line1[0], line1_result, resultRGB1, 1)
show_result(peaks_line2[0], line2_result, resultRGB2, 2)

diff_status1 = get_diff_status(line1_result, mean_size, bias)
diff_status2 = get_diff_status(line2_result, mean_size, bias)
print('result status:')
print(diff_status1)
print(diff_status2)

'''
#  保存模板
cv2.destroyAllWindows()
oooooo = grayline1od
cv2.imshow('mould', oooooo)
mould = oooooo[7:25,169:179]
cv2.imshow('result', mould)
# mould = np.uint8(mould)
cv2.imwrite("/home/py/PycharmProjects/mould0/877_1_1_1.jpg", mould, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
'''

# plt.plot(col_sum_line1d, 'b')
# plt.plot(col_sum_line1,'r')
# plt.show()





cv2.waitKey(0)
cv2.destroyAllWindows()
