# -*- coding: utf-8 -*


import numpy as np
import cv2
import os
from pylab import *
from matplotlib import pyplot as plt
import scipy.signal as signal
# from skimage import data, draw, color, transform, feature
import detect_peaks

# mould detect
def mould_detect(img2, template, methods, weight):
    ww, hh = template.shape[::-1]
    img = img2.copy()
    imgf = img2.copy()
    method = eval(methods)
    res = cv2.matchTemplate(img, template, method)
    # print('mould detect result:')
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
    print('number of result')
    print(i)

    loc_result = []
    for x, y in loc_choice:
        v = res[y, x]
        loc_result.append((x, y, v))

    return loc_result



def detect_number(img2):
    tempNum = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # tempWeight = [0.30, 0.18, 0.25, 0.25, 0.25, 0.25, 0.3, 0.2, 0.35, 0.3]
    tempWeight = [0.23] * 10    #  here here!!!
    # tempWeight = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    # tempNum = ['0', '6', '8', '9']
    # tempWeight = [0.25, 0.26, 0.28, 0.26]
    # tempNum = ['0','8']
    # tempNum = ['0', '1']
    line_number_loc = []

    for num, weight in zip(tempNum, tempWeight):
        line_number_loc0 = []
        template1 = cv2.imread("./mould/{}/001.jpg".format(num), 0)
        template2 = cv2.imread("./mould/{}/002.jpg".format(num), 0)
        template3 = cv2.imread("./mould/{}/003.jpg".format(num), 0)
        template_list = ['template1', 'template2', 'template3']
        methods = 'cv2.TM_SQDIFF_NORMED'
        for temp in template_list:
            template = eval(temp)
            template = cv2.equalizeHist(template)
            print('current bumber:')
            print(num)
            loc_result= mould_detect(img2, template, methods, weight)
            # line_number_loc0.append(loc_result)
            line_number_loc0 += loc_result


        line_number_loc += line_number_loc0



    return line_number_loc



def result_fliter(line_number_loc):

    whb = 2
    ww = 12 - whb
    hh = 18 - whb
    bias = 3
    loc_fliter = []

    for x, y, v in line_number_loc:
        flag = 0
        for i, j, v2 in loc_fliter:
            if abs(x-i) < bias and abs(y-j) < bias:
                flag = 1
            elif x - i < ww and y - j < hh and x>i and j>y:
                if v < v2:
                    loc_fliter.remove((i, j, v2))
                    flag = 0
                    break

                else:
                    flag = 1
                    break
            elif i - x < ww and j - y < hh and i>x and y>j:
                if v < v2:
                    loc_fliter.remove((i, j, v2))
                    flag = 0
                    break

                else:
                    flag = 1
                    break
            elif i - x < ww and y - j < hh and i>x and j>y:
                if v < v2:
                    loc_fliter.remove((i, j, v2))
                    flag = 0
                    break

                else:
                    flag = 1
                    break
            elif x - i < ww and j - y < hh and x>i and y>j:
                if v < v2:
                    loc_fliter.remove((i, j, v2))
                    flag = 0
                    break

                else:
                    flag = 1
                    break


        if flag == 0:
            # print((x,y))
            loc_fliter.append((x,y,v))


    return loc_fliter



def get_result_of_mould(image):

    line_number_loc = detect_number(image)
    print(line_number_loc)
    loc_fliter = result_fliter(line_number_loc)

    mould_reult = []
    for x,y,z in loc_fliter:
        mould_reult.append((x,y,12,18))

    return mould_reult


if __name__ == '__main__':
    bann = '000003'
    src = "./extract/{}.jpg".format(bann)
    RGB = cv2.imread(src)
    gray = cv2.cvtColor(RGB, cv2.COLOR_BGR2GRAY)

    mould_reult = get_result_of_mould(gray)

    print(mould_reult)

    for x, y, w, h in mould_reult:
        cv2.rectangle(gray, (x, y), (x + w, y + h), 255, 2)

    plt.imshow(gray, cmap="gray")
    plt.title('Result'), plt.xticks([]), plt.yticks([])

    plt.show()
