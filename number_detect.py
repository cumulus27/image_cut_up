import numpy as np
import cv2
from pylab import *
from matplotlib import pyplot as plt
import scipy.signal as signal

# src="/home/py/PycharmProjects/image_process/extract/000048.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000027.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000019.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000118.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000033.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000055.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000158.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000724.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000063.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000877.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000067.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000150.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000003.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000018.jpg"
src="/home/py/PycharmProjects/image_process/extract/000041.jpg"
# src="/home/py/PycharmProjects/image_process/extract/000097.jpg"

def hist_plt_plot(b,g,r):
    histImgB = cv2.calcHist([b],[0],None,[256],[1,256])
    histImgG = cv2.calcHist([g],[0],None,[256],[1,256])
    histImgR = cv2.calcHist([r],[0],None,[256],[1,256])
    plt.plot(histImgB,'b')
    #plt.show()    
    plt.plot(histImgG,'g')
    #plt.show()
    plt.plot(histImgR,'r')
    plt.show()


RGB=cv2.imread(src)
cv2.imshow('RGB',RGB)
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
per = 0.96
xs = r.shape[0]
ys = r.shape[1]
th_po = xs*ys*per
th_po = int(th_po)

gray=cv2.cvtColor(RGB,cv2.COLOR_BGR2GRAY)
graysort = np.sort(gray, axis=None)

#rsort = np.sort(r, axis=None)
#gsort = np.sort(g, axis=None)
#bsort = np.sort(b, axis=None)
#rth = rsort[th_po]
#gth = gsort[th_po]
#bth = bsort[th_po]

# rmean = np.mean(rsort[th_po:])
# gmean = np.mean(gsort[th_po:])
# bmean = np.mean(bsort[th_po:])

th_val = graysort[th_po]
rmean = np.mean(r[np.where(gray>th_val)])
bmean = np.mean(b[np.where(gray>th_val)])
gmean = np.mean(g[np.where(gray>th_val)])

kb = (rmean + gmean + bmean) / (3 * bmean)
kr = (rmean + gmean + bmean) / (3 * rmean)
kg = (rmean + gmean + bmean) / (3 * gmean)
br = r * kr
bg = g * kg
bb = b * kb
 
balRGB = cv2.merge([bb, bg, br])
balRGB = np.uint8(balRGB)
cv2.imshow('BalRGB ',balRGB)
balRGB = cv2.medianBlur(balRGB,3)
cv2.imshow('mediaBalRGB ',balRGB)


# b, g, r = cv2.split(balRGB)



# color weight  RGB
b, g, r = cv2.split(balRGB)
b=np.int16(b)
g=np.int16(g)
r=np.int16(r)
rweight = abs(r-b) + abs(r-g)+ abs(b-g)
# rweight = (r-b)**2 + (r-g)**2+ (b-g)**2
rsum = r+b+g
# fuzzy = (rsum*2  - rweight)/rsum
# fuzzy = (255 - rweight)/255
fuzzy = 1 / (rweight/rsum + 1)



tags = np.ones(r.shape)
tags = tags * 255 * fuzzy 
tags = np.uint8(tags)
cv2.imshow('tags ', tags)

rf = r * fuzzy
rf = np.uint8(rf)
gf = g * fuzzy
gf = np.uint8(gf)
bf = b * fuzzy
bf = np.uint8(bf)

# rf = cv2.medianBlur(rf,3)
# gf = cv2.medianBlur(gf,3)
# bf = cv2.medianBlur(bf,3)
fRGB = cv2.merge([bf, gf, rf])

# fRGB = cv2.GaussianBlur(fRGB,(3,3),0.5)
fRGB = cv2.medianBlur(fRGB,3)
cv2.imshow('fRGB',fRGB)
b=np.uint8(b)
g=np.uint8(g)
r=np.uint8(r)

b, g, r = cv2.split(fRGB)

'''
# color weight
# fuzzy = np.zeros(r.shape)
Window_size = 1
xmax = len(fuzzy)
ymax = len(fuzzy[0])

for i in range(xmax):
    for j in range(ymax):
        """
        rweight = 0
        rsum = 0
        for k in range(max(i-Window_size, 0),min(i+Window_size, xmax)):
            for n in range(max(j-Window_size, 0),min(j+Window_size, ymax)):
                alpha = max(0, Window_size-(abs(k-i) + abs(n-j)) + 1)
                rweight = rweight + alpha * (abs(r[k][n]-b[k][n]) + abs(g[k][n]-b[k][n]) + abs(r[k][n]-g[k][n]))
                rsum = rsum + alpha * r[k][n]+b[k][n]+g[k][n]

        fuzzy[i][j] = rweight*rsum
        """
        rweight = (r[i][j] - b[i][j])**2 + (g[i][j] - b[i][j])**2 + (r[i][j] - g[i][j])**2
        rsum = (r[i][j] + b[i][j] + g[i][j])/(255*3)
        fuzzy[i][j] = rsum / (rweight+1)

fuzzy = fuzzy / (Window_size**2)

fuzim = np.uint8(fuzzy)
cv2.imshow('fuzim',fuzim)
'''

'''
sobelx = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=1)
sobely = cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=1)
sobelx = np.uint8(np.absolute(sobelx))
sobely = np.uint8(np.absolute(sobely))
sobelcombine = cv2.bitwise_or(sobelx,sobely)
cv2.imshow('sobelcombine',sobelcombine )
'''

'''
# zhi fang tu jun heng 
Histr = cv2.equalizeHist(r)
Histg = cv2.equalizeHist(g)
Histb = cv2.equalizeHist(b)
HistRGB = cv2.merge([Histb, Histg, Histr])
# HistRGB = cv2.GaussianBlur(HistRGB,(3,3),0.5)
cv2.imshow('HistRGB0',HistRGB)

# fen kuai jun heng
blockxnum = 20
blockynum = 1
blockx = r.shape[1] // blockxnum
blocky = r.shape[0] // blockynum

Histr = np.zeros(r.shape)
Histb = np.zeros(b.shape)
Histg = np.zeros(g.shape)
for i in range(blockxnum):
    for j in range(blockynum):
        fpx = i * blockx
        fpy = j * blocky
        if i == blockxnum -1:
            ppx = r.shape[1]
        else:
            ppx = (i+1) * blockx
        if j == blockynum -1:
            ppy = r.shape[0]
        else:            
            ppy = (j+1) * blocky
    
        Histr[fpy:ppy,fpx:ppx] = cv2.equalizeHist(r[fpy:ppy,fpx:ppx] )
        Histg[fpy:ppy,fpx:ppx] = cv2.equalizeHist(g[fpy:ppy,fpx:ppx] )
        Histb[fpy:ppy,fpx:ppx] = cv2.equalizeHist(b[fpy:ppy,fpx:ppx] )
    
HistRGB = cv2.merge([Histb, Histg, Histr])
HistRGB = np.uint8(HistRGB)
# HistRGB = cv2.GaussianBlur(HistRGB,(3,3),0.5)
cv2.imshow('HistRGB1',HistRGB)
'''

'''  
cv2.imshow('Histr',Histr)
cv2.imshow('Histg',Histg)
cv2.imshow('Histb',Histb)
'''

'''
hist_plt_plot(b,g,r)
hist_plt_plot(Histb ,Histg ,Histr)
'''

'''
r = Histr
r = cv2.GaussianBlur(r,(3,3),0.5)
g = Histg
g = cv2.GaussianBlur(g,(3,3),0.5)
b = Histb
b = cv2.GaussianBlur(b,(3,3),0.5)
'''


# (h, w) = fRGB.shape[:2]
LAB=cv2.cvtColor(fRGB,cv2.COLOR_BGR2LAB)
cv2.imshow('LAB',LAB)
cv2.imshow('labL',LAB[:,:,0])
# cv2.imshow('labA',LAB[:,:,1])
# cv2.imshow('labB',LAB[:,:,2])
V=LAB[:,:,0]
cv2.imshow('L',V)
V=cv2.equalizeHist(V)

'''
# 求行和列和  圈数字位置法
row_sum = np.sum(V,axis=1)
col_sum = np.sum(V,axis=0)
# row_dev = np.diff(np.int64(row_sum),n=5)
# col_dev = np.diff(np.int64(col_sum),n=5)
row_dev = abs(np.gradient(np.int64(row_sum)))
col_dev = abs(np.gradient(np.int64(col_sum)))
row_diff = np.diff(row_dev)
col_diff = np.diff(col_dev)

# plt.plot(row_sum,'b')
# plt.plot(col_sum,'r')
# plt.plot(col_dev,'g')
# plt.plot(row_dev,'y')
# plt.show()

row_li = np.where(row_sum > 2000)[0]
col_li = np.where(col_sum > 2000)[0]
# print(row_li)
# print(col_li)


row_bd = np.where(row_diff < 0)[0]
col_bd = np.where(col_diff < 0)[0]
# print(row_bd)
# print(col_bd)

row_bd = row_bd[np.where(row_bd > row_li[0])]
row_bd = row_bd[np.where(row_bd < row_li[-1])]
col_bd = col_bd[np.where(col_bd > col_li[0])]
col_bd = col_bd[np.where(col_bd < col_li[-1])]

# print(row_bd)
# print(col_bd)
'''

# 高亮像素计数圈数字位置法
vhigh = V
vhigh[np.where(V < 140)] = 0
row_sum = np.sum(vhigh,axis=1)
col_sum = np.sum(vhigh,axis=0)

row_bd = np.where(row_sum > 2200)[0]
col_bd = np.where(col_sum > 800)[0]



print(row_bd)
print(col_bd)
# plt.plot(row_sum,'b')
# plt.plot(col_sum,'r')
# plt.show()
cps = 4

row_f = max(0, row_bd[0]-cps)
row_l = min(row_bd[-1]+cps, fRGB.shape[0])
col_f = max(0, col_bd[0]-cps)
col_l = min(col_bd[-1]+cps, fRGB.shape[1])

nbRGB = fRGB[row_f:row_l,col_f:col_l,:]
resultRGB = RGB[row_f:row_l,col_f:col_l,:]

cv2.imshow('nbRGB', nbRGB)
LAB=cv2.cvtColor(nbRGB,cv2.COLOR_BGR2LAB)
V=LAB[:,:,0]
cv2.imshow('L',V)

(h, w) = nbRGB.shape[:2]

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

# HistV=cv2.equalizeHist(V)
LAB[:,:,0]=HistV
histRBG=cv2.cvtColor(LAB, cv2.COLOR_LAB2BGR)
gray=cv2.cvtColor(histRBG,cv2.COLOR_BGR2GRAY)


'''
# 图像大小调整
(h_new, w_new) = gray.shape
srot = np.zeros((h_new, w_new,3))
srot[h_new // 2 - 24:h_new // 2 + 24, :,:] = 1
srot=np.uint8(srot)
gray = gray * srot
RGB_new=np.zeros((48,w_new,3))
for i in range (0,48):
    RGB_new[i,:]=gray[h_new // 2 - 24+i, :]
RGB_new = np.uint8(RGB_new)
cv2.imshow('RGB_new',RGB_new)
#np.delete(HistRGB, np.where(HistRGB[:,:,0]==0 and HistRGB[:,:,1]==0 and HistRGB[:,:,2]==0))
#cv2.imshow('HistRGB11',HistRGB)

#cv2.imshow("histImgB", histImgB)
#cv2.imshow("histImgG", histImgG)
#cv2.imshow("histImgR", histImgR)
'''


'''
# color weight HSV
HSV=cv2.cvtColor(RGB,cv2.COLOR_BGR2HSV)
imh, ims, imv = cv2.split(HSV)
cv2.imshow('hsvh',imh)
cv2.imshow('hsvs',ims)
cv2.imshow('hsvv',imv)
'''

'''
# 分通道边缘检测
lapr = cv2.Laplacian(r, cv2.CV_64F,ksize=5)
lapr = np.uint8(np.absolute(lapr))
lapg = cv2.Laplacian(g, cv2.CV_64F,ksize=5)
lapg = np.uint8(np.absolute(lapg))
lapb = cv2.Laplacian(b, cv2.CV_64F,ksize=5)
lapb = np.uint8(np.absolute(lapb))
lapRGB = cv2.merge([lapb, lapg, lapr])
lapRGB = cv2.GaussianBlur(lapRGB,(3,3),0.1)
cv2.imshow('lapRGB',lapRGB)
'''

gaussGray = cv2.GaussianBlur(gray,(3,3),1)
lapGray = cv2.Laplacian(gaussGray, cv2.CV_64F,ksize=3)
lapGray = np.uint8(lapGray)
# lapGray = np.uint8(np.absolute(lapGray))
cv2.imshow('lapGray',lapGray)
# histGray = cv2.cvtColor(HistRGB,cv2.COLOR_BGR2GRAY)
histGray = gray
lhGray = cv2.add(lapGray*0.1, histGray*0.9)
lhGray = np.uint8(lhGray)
cv2.imshow('lapGray', lapGray)
cv2.imshow('histGray', histGray)
cv2.imshow('lhGray', lhGray)


'''
# 膨胀 腐蚀
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))  
dilated = cv2.dilate(lhGray,kernel)  
#显示膨胀后的图像  
eroded = cv2.erode(dilated,kernel)  
#显示腐蚀后的图像  
cv2.imshow("Eroded Image",eroded);
'''

# 像素统计分割法
# simpleV=cv2.equalizeHist(V)
# cv2.imshow('simpleV', simpleV)
# row_sumt = np.sum(simlpeV,axis=1)
# plt.plot(row_sumt,'r')

row_sum2 = np.sum(gray,axis=1)
# row_diff2 = np.diff(np.int64(row_sum2))
x0 = gray.shape[0] // 2
row_sump = row_sum2[x0-5:x0+6]
row_bd2 = np.where(row_sum2 == np.min(row_sump))[0][0] #隐藏bug 万一有多个值
# plt.plot(row_sum2,'b')
# plt.show()

# 截取第一行和第二行
rdt = 1
grayline1 = gray[0:row_bd2+rdt+1,:]
grayline2 = gray[row_bd2-rdt:,:]

resultRGB1 = resultRGB[0:row_bd2+rdt+1,:]
resultRGB2 = resultRGB[row_bd2-rdt:,:]
# graylines1 = simpleV[0:row_bd2+rdt+1,:]
# graylines2 = simpleV[row_bd2-rdt:,:]
cv2.imshow('grayline1',grayline1)
cv2.imshow('grayline2',grayline2)
# cv2.imshow('graylines1',graylines1)
# cv2.imshow('graylines2',graylines2)
'''
grayhigh1 = grayline1
grayhigh2 = grayline2
grayhigh1[np.where(grayline1 < 180)] = 0
grayhigh2[np.where(grayline2 < 180)] = 0

col_sh1 = np.sum(grayhigh1,axis=0)
col_sh2 = np.sum(grayhigh2,axis=0)

# plt.plot(col_sh1,'b')
# plt.plot(col_sh2,'r')
# plt.show()


col_bd1 = np.where(col_sh1 > 250)[0]
col_bd2 = np.where(col_sh2 > 250)[0]
# print(col_bd1)
# print(col_bd2)


cps = 1
# print(grayline1.shape)
grayline1 = grayline1[:,col_bd1[0]-cps:col_bd1[-1]+cps]
grayline2 = grayline2[:,col_bd2[0]-cps:col_bd2[-1]+cps]
resultRGB1 = resultRGB1[:,col_bd1[0]-cps:col_bd1[-1]+cps]
resultRGB2 = resultRGB2[:,col_bd2[0]-cps:col_bd2[-1]+cps]
# print(grayline1.shape)
'''
# 每一行计算列和
# col_sum_line1 = np.sum(grayline1,axis=0)
# col_sum_line2 = np.sum(grayline2,axis=0)
# 计算平方和
# col_sum_line1 = np.sum(grayline1**2,axis=0)
# col_sum_line2 = np.sum(grayline2**2,axis=0)

# 求和部分加入阈值
shd = 180
grayline1hd = grayline1
grayline2hd = grayline2
grayline1hd[np.where(grayline1 < shd)] //= 2
grayline2hd[np.where(grayline2 < shd)] //= 2
col_sum_line1 = np.sum(grayline1hd,axis=0)
col_sum_line2 = np.sum(grayline2hd,axis=0)


# plt.plot(col_sum_line1,'b')
# plt.show()


#滤波
ksize = 3
col_sum_line1 = signal.medfilt(col_sum_line1,ksize)
col_sum_line2 = signal.medfilt(col_sum_line2,ksize)

col_sum_line1d = np.max(col_sum_line1) - col_sum_line1
col_sum_line2d = np.max(col_sum_line2) - col_sum_line2
# plt.plot(col_sum_line1d,'r')

peaks_line1 = signal.find_peaks_cwt(col_sum_line1d, np.arange(1,9))
peaks_line2 = signal.find_peaks_cwt(col_sum_line2d, np.arange(1,9))
peaks_line1 = np.concatenate((np.array([0]),peaks_line1,np.array([len(col_sum_line1d)-1])), axis=0)
peaks_line2 = np.concatenate((np.array([0]),peaks_line2,np.array([len(col_sum_line2d)-1])), axis=0)

# peaks_high1 = col_sum_line1d[peaks_line1]
# peaks_high2 = col_sum_line1d[peaks_line2]

peaks_diff1 = np.diff(peaks_line1)
peaks_diff2 = np.diff(peaks_line2)

# 中位数做参考值
diff_queue = np.append(peaks_diff1, peaks_diff2)
diff_queue = diff_queue[np.where(diff_queue > 8)]
diff_queue = diff_queue[np.where(diff_queue < 16)]
mean_size = np.median(diff_queue)

# 众数做参考值
# counts = np.bincount(diff_queue)
# mean_size = np.argmax(counts)

trust1 = np.zeros(peaks_line1.shape)
trust2 = np.zeros(peaks_line2.shape)

num = 16
# mean_size = round(gray.shape[1]/num)
print('参考宽度：')
print(mean_size)
bias = round(mean_size*0.22)


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

def get_neighber_trust(trust, peaks_diff, diff_status):
    neighbers = np.zeros(trust.shape)
    n = len(peaks_diff)
    for i in range(n):
        if diff_status[i] == 0:
            neighbers[i] += 1
            neighbers[i+1] += 1

    trust = neighbers*0.4 + trust
    return trust



trust1 = get_neighber_trust(trust1, peaks_diff1, diff_status1)
trust2 = get_neighber_trust(trust2, peaks_diff2, diff_status2)

print('First Trust:')
print(trust1)
print(trust2)

def handle_small_diff(trust, peaks_diff, diff_status, mean_size, bias):
    n = len(peaks_diff)
    for i in range(n):
        if diff_status[i] == 1:
            if i < n-1 and diff_status[i+1] == 1:
                if peaks_diff[i + 1] + peaks_diff[i] < mean_size + bias+1:
                    peaks_diff[i + 1] = peaks_diff[i + 1] + peaks_diff[i]
                    peaks_diff[i] = 0
                    diff_status = get_diff_status(peaks_diff, mean_size, bias)
                    trust[i] = -1
            elif i >= n-1:
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
            elif i == 0 and peaks_diff[i] < bias + 1:
                peaks_diff[i + 1] = peaks_diff[i + 1] + peaks_diff[i]
                peaks_diff[i] = 0
                diff_status = get_diff_status(peaks_diff, mean_size, bias)
                trust[i] = -4
            elif i > 0 and diff_status[i-1] == 0:
                # print(diff_status[i])
                # print(diff_status[i+1])
                if diff_status[i+1] == 2:
                    peaks_diff[i + 1] = peaks_diff[i + 1] + peaks_diff[i]
                    peaks_diff[i] = 0
                    diff_status = get_diff_status(peaks_diff, mean_size, bias)
                    trust[i] = -5
                elif diff_status[i+1] == 0:
                    if diff_status[i] < bias + 1:
                        # 超小块噪声 直接归并到旁边
                        if peaks_diff[i-1] > peaks_diff[i+1]:
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
            elif diff_status[i+1] == 0:
                if i > 0 and diff_status[i-1] == 2:
                    peaks_diff[i] = peaks_diff[i - 1] + peaks_diff[i]
                    peaks_diff[i-1] = 0
                    diff_status = get_diff_status(peaks_diff, mean_size, bias)
                    trust[i-1] = -9
            elif diff_status[i+1] == 2 and diff_status[i-1] != 2:
                peaks_diff[i + 1] = peaks_diff[i + 1] + peaks_diff[i]
                peaks_diff[i] = 0
                diff_status = get_diff_status(peaks_diff, mean_size, bias)
                trust[i] = -12
            elif diff_status[i+1] == 2 and diff_status[i-1] == 2:
                trust[i] = -10
                print('warning!!!两边都是大区，不知道该划入哪边')
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
                fenjie = round(diff/j)
                # print(fenjie)
                if fenjie < ll and fenjie > sl+1:
                    # print(peaks_diff)
                    peaks_diff[i] = fenjie
                    for n in range(j-2):
                        peaks_diff = np.insert(peaks_diff,i,fenjie)
                        i += 1
                    # print(peaks_diff)
                    peaks_diff = np.insert(peaks_diff, i, diff - (j-1)*fenjie)
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

print('result:')
print(line1_result)
print(line2_result)


# 两个连续偏大情况纠错
def three2two_err_fix(line_result, num, mean_size, bias):
    length = len(line_result)
    too_big = np.zeros(line_result.shape)
    too_big = np.append(too_big,too_big)
    if length >= num:
        return line_result
    elif length < num:
        i = 0
        for result in line_result:
            if result > mean_size + 2:
                # print(i)
                too_big[i] = 1
                # if i == 0:             # 首项不处理  暂时处理方法  前面搞定后去掉
                #     too_big[i] = 0
                if i > 0 and too_big[i-1] == 1:
                    mix = result + line_result[i - 1]
                    sl = mean_size - bias
                    ll = mean_size + bias
                    trsplit = round(mix/3)
                    if trsplit > sl and trsplit < ll:
                        line_result[i-1] = trsplit
                        line_result[i] = trsplit
                        line_result = np.insert(line_result, i, mix - 2 * trsplit)
                        i += 1
                        too_big[i] = 2
                    elif trsplit > ll:
                        print('err! 分成三个还是太大，请检查')
            i += 1
    return line_result


line1_result = three2two_err_fix(line1_result, num, mean_size, bias)
line2_result = three2two_err_fix(line2_result, num, mean_size, bias)

print('修正3to2bug后：')
print(line1_result)
print(line2_result)

# 连续两个一个偏大一个偏小情况纠正
diff_status1 = get_diff_status(line1_result, mean_size, bias)
diff_status2 = get_diff_status(line2_result, mean_size, bias)
print('result status:')
print(diff_status1)
print(diff_status2)


def big_small_fix(line_result, diff_status):
    for i, status in enumerate(diff_status):
        if i > 0 and ((status == 1 and diff_status[i - 1] == 2) or (status == 2 and diff_status[i - 1] == 1)):
            if i < diff_status.shape[0] - 1 and diff_status[i+1] != 0:
                print('warning!!! 连续三个位置异常，处理可能出错')

            mix = line_result[i-1] + line_result[i]
            line_result[i - 1] = mix // 2
            line_result[i] = mix - line_result[i - 1]
            diff_status[i] = -1

    return line_result


line1_result = big_small_fix(line1_result, diff_status1)
line2_result = big_small_fix(line2_result, diff_status2)

print('修正big_small bug后：')
print(line1_result)
print(line2_result)


def show_result(first, line_result, resultRGB, line):
    point = first
    name = 1
    fat = 2
    for i in line_result:
        sigleim = resultRGB[:,max(point-fat,0):min(point+i+fat,resultRGB.shape[1]-1)]
        cv2.imshow(str(line)+'sigle'+str(name),sigleim)
        point += i
        name += 1


show_result(peaks_line1[0], line1_result, resultRGB1, 1)
show_result(peaks_line2[0], line2_result, resultRGB2, 2)

diff_status1 = get_diff_status(line1_result, mean_size, bias)
diff_status2 = get_diff_status(line2_result, mean_size, bias)
print('result status:')
print(diff_status1)
print(diff_status2)

plt.plot(col_sum_line1d,'b')
plt.show()


"""
# mser
mser = cv2.MSER_create(_min_area=40, _max_area=90)
regions, boxes = mser.detectRegions(gray)

gray = lhGray

temp1 = np.zeros(gray.shape)
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
temp1=temp1*gray
temp1=np.uint8(temp1)
cv2.imshow('temp1',temp1)
xn=np.array(xn)
yn=np.array(yn)
x1=xn[np.where(yn<min(yn)+hmin)]
y1=yn[np.where(yn<min(yn)+hmin)]
x2=xn[np.where(yn>min(yn)+hmax)]
y2=yn[np.where(yn>min(yn)+hmax)]

x1_left=min(x1)
y1_left=y1[np.where(x1 == x1_left)][0]
x1_right = max(x1)
y1_right = y1[np.where(x1 == x1_right)][0]
k = (y1_right-y1_left)/(x1_right-x1_left)

y1_mid = (y1_left + y1_right)//2
y_line = y1_mid+hmid  
line1 = np.zeros((h, w))
line1[y1_mid-3:y_line+2, :]=1
numin1=line1*gray
numin1 = np.uint8(numin1)
print(x1)
print(y1)
# 3维度聚类
# num1=np.zeros((h,w,3))
# ret1, numarea = cv2.threshold(numin1,200,1,cv2.THRESH_BINARY)
#
# numarea=np.uint8(numarea)
# # cv2.imshow('numa',numarea)
# num1[:,:,0]=numarea
# num1[:,:,1]=numarea
# num1[:,:,2]=numarea
# A1=A*num1
# A1 = np.uint8(A1)
# # cv2.imshow('AAA',A1)
# blue = A1[:,:,0]
# barea=blue[np.where(blue!=0)]
# b_MEAN=np.mean(barea)
# green = A1[:,:,1]
# garea=green[np.where(green!=0)]
# g_MEAN=np.mean(garea)
# red = A1[:,:,2]
# rarea=red[np.where(red!=0)]
# r_MEAN=np.mean(rarea)

# numaaa=np.zeros((h,w))
# MEAN_color = np.array([b_MEAN, g_MEAN, r_MEAN])
# for i in range(0, h):
#     for j in range(0, w):
#         difference=np.abs(A[i,j,:]-MEAN_color)
#         if difference[0]<20 and difference[1]<20 and difference[2]<20:
#             numaaa[i,j]=1
#
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
# closed = cv2.morphologyEx(numaaa, cv2.MORPH_CLOSE, kernel)
# numb=np.zeros((h,w,3))
# # numb[:,:,0]=closed
# # numb[:,:,1]=closed
# # numb[:,:,2]=closed
# numb[:,:,0]=numaaa
# numb[:,:,1]=numaaa
# numb[:,:,2]=numaaa
# tmp = A*numb
# tmp = np.uint8(tmp)
# cv2.imshow('tmp', numaaa)
# cv2.waitKey()

# 抠数字部分
angle= rad2deg(arctan(k))
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(numin1, M, (w, h), borderValue=(0))
rotated=line1*rotated
rotated = np.uint8(rotated)
cv2.imshow('rot', rotated)

# 这里的200到时候通过直方图改阈值,同时为了考虑明暗，试着用局部阈值
# for i in range(0, h):
#     for j in range(0, w):
#         if rotated[i, j] < 220:
#             rotated[i, j] = 0
# rotated = cv2.adaptiveThreshold(rotated,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,0)

ret, rotated = cv2.threshold(rotated,150,255,cv2.THRESH_BINARY)

'''
# Fuzzy clustering
fuzzy = np.zeros(rotated.shape)
Window_size = 2
xmax = len(fuzzy)
ymax = len(fuzzy[0])
for i in range(xmax):
    for j in range(ymax):
        rsum = 0
        for k in range(max(i-Window_size, 0),min(i+Window_size, xmax)):
            for n in range(max(j-Window_size, 0),min(j+Window_size, ymax)):
                alpha = max(0, Window_size-(abs(k-i) + abs(n-j)) + 1)
                rsum = rsum + alpha * rotated[k][n]

        fuzzy[i][j] = rsum

fuzzy = fuzzy / Window_size**2
fuzim = np.uint8(fuzzy)
cv2.imshow('fuzim',fuzim)
ret, rotated = cv2.threshold(fuzim,180,255,cv2.THRESH_BINARY)
'''

cv2.imshow('rot2', rotated)

column_sum = np.sum(rotated, 0)
xline1=np.zeros((1,0))
for i in range(0, len(column_sum)-1):
    if column_sum[i]<100 and column_sum[i+1]:
        xline1=np.append(xline1, (i+1))
xline1 = np.append(xline1,xline1[-1]+wmid)
xline1=np.uint8(xline1)

'''
for i in range(0,len(xline1)-1):
    temp = zeros((h, w))
    tmp = zeros((h, w, 3))
    temp[y1_mid-3:y_line+2,xline1[i]:xline1[i+1]] = 1
    tmp[:,:,0]=temp
    tmp[:,:,1]=temp
    tmp[:,:,2]=temp

    # 这里不该是乘以A，因为A的第一行没旋转
    tmp=tmp*RGB
    temp = temp*rotated
    temp = np.uint8(temp)
    tmp=np.uint8(tmp)
    cv2.imshow('gray1', temp)
    cv2.imshow('gray2', gray)
    cv2.imshow('rgb',tmp)
    cv2.waitKey()
'''


# while min(x1)>14:
#     x1=np.append(x1,min(x1)-wmid)
#
#     y1=np.append(y1,np.uint8(sum(y1)/len(y1)))
#
# while max(x1)<220:
#     x1 = np.append(x1, max(x1) + wmid)
#     y1 = np.append(y1, np.uint8(sum(y1) / len(y1)))
#
# while min(x2)>14:
#     x2=np.append(x2,min(x2)-wmid)
#
#     y2=np.append(y2,np.uint8(sum(y2)/len(y2)))
#
# while max(x2)<220:
#     x2 = np.append(x2, max(x2) + wmid)
#     y2 = np.append(y2, np.uint8(sum(y2) / len(y2)))
#
#
# for i in range(0,len(x1)):
#     temp[y1[i]:y1[i]+hmid,x1[i]:x1[i]+wmid]=1
#
# for i in range(0,len(x2)):
#     temp[y2[i]:y2[i]+hmid,x2[i]:x2[i]+wmid]=1





#测试数字百分比，大概在17%到38%
# k=0
# for i in range(y1[1],y1[1]+hmid):
#     for j in range(x1[1],x1[1]+wmid):
#         if temp[i,j]>180:
#             print(i,j)
#             k=k+1
# print(k/(hmid*wmid))

# cv2.imshow('A', A)
# cv2.imshow('gray1', temp)
# cv2.imshow('gray2', gray)
# cv2.waitKey()


"""
cv2.waitKey(0)
cv2.destroyAllWindows()
