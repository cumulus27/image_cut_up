import numpy as np
import cv2
import os
from pylab import *
from matplotlib import pyplot as plt
import scipy.signal as signal
# from skimage import data, draw, color, transform, feature
import detect_peaks

# 模板匹配
def mould_detect(img2, template, methods, weight):
    ww, hh = template.shape[::-1]
    img = img2.copy()
    imgf = img2.copy()
    method = eval(methods)
    res = cv2.matchTemplate(img, template, method)
    # print('模板匹配结果：')
    # print(res)
    loc_choice = []

    i = 0
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        threshold = weight
        loc = np.where(res <= threshold)
        for pt in zip(*loc[::-1]):
            loc_choice.append(pt)
            i += 1
            cv2.rectangle(img, pt, (pt[0] + ww, pt[1] + hh), 255, 2)
            # while
    else:
        threshold = 1 - weight
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            loc_choice.append(pt)
            i += 1
            cv2.rectangle(img, pt, (pt[0] + ww, pt[1] + hh), 255, 2)

    print(methods)
    print('个数：')
    print(i)

    loc_result = []
    for x, y in loc_choice:
        v = res[y, x]
        loc_result.append((x, y, v))

    return loc_result



def detect_number(img2):
    tempNum = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    tempWeight = [0.30, 0.18, 0.25, 0.25, 0.25, 0.25, 0.3, 0.2, 0.35, 0.3]
    # tempWeight = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    # tempNum = ['0', '6', '8', '9']
    # tempWeight = [0.25, 0.26, 0.28, 0.26]
    # tempNum = ['0','8']
    # tempNum = ['0', '1']
    line_number_loc = []

    for num, weight in zip(tempNum, tempWeight):
        line_number_loc0 = []
        template1 = cv2.imread("/home/py/PycharmProjects/image_cut_up/mould/{}/001.jpg".format(num), 0)
        template2 = cv2.imread("/home/py/PycharmProjects/image_cut_up/mould/{}/002.jpg".format(num), 0)
        template3 = cv2.imread("/home/py/PycharmProjects/image_cut_up/mould/{}/003.jpg".format(num), 0)
        template_list = ['template1', 'template2', 'template3']
        methods = 'cv2.TM_SQDIFF_NORMED'
        for temp in template_list:
            template = eval(temp)
            template = cv2.equalizeHist(template)  # 模板直方图均衡
            print('当前数字：')
            print(num)
            loc_result= mould_detect(img2, template, methods, weight)
            line_number_loc0.append(loc_result)


        line_number_loc += line_number_loc0



    return line_number_loc



def result_fliter(line_number_loc):

    loc_fliter = []

    for x, y, v in line_number_loc:
        flag
        for i, j, v2 in loc_fliter:
            pass

