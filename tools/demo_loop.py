import argparse
import tools.find_mxnet
import mxnet as mx
import os
import sys
from detect.detector import Detector
from symbol.symbol_factory import get_symbol
from skimage import io
import cv2
import numpy as np
import rot
from pylab import *
from demo import *


if __name__ == '__main__':
    # Input image

    args = parse_args()
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu_id)

    # assert len(image_list) > 0, "No valid image specified to detect"

    network = None if args.deploy_net else args.network
    class_names = parse_class_names(args.class_names)
    if args.prefix.endswith('_'):
        prefix = args.prefix + args.network + '_' + str(args.data_shape)
    else:
        prefix = args.prefix
    detector = get_detector(network, prefix, args.epoch,
                            args.data_shape,
                            (args.mean_r, args.mean_g, args.mean_b),
                            ctx, len(class_names), args.nms_thresh, args.force_nms)

    for i in range(439,932):
        bann = str(i + 1)
        while len(bann) < 6:
            bann = '0' + bann

        print(bann)
        # src = "./out/{}.jpg".format(bann)
        # if os.path.exists(src):
        #     RGB = cv2.imread(src)
        # else:
        #     print('File not exists!!')
        #     continue

        # img_path = './data/VOCdevkit/VOC2007/test/'
        # simdir = os.listdir(img_path)
        # image_list = []
        # for i in range(len(simdir)):
        #     image_list.append(img_path + simdir[i])
        # parse image list
        # image_list = [i.strip() for i in args.images.split(',')]
        image_list = './data/VOCdevkit/VOC2007/test/{}.jpg'.format(bann)
        if not os.path.exists(image_list):
            print('File not exists!!')
            continue

        imgg = cv2.imread(image_list)
        imgg[:, :, (0, 1, 2)] = imgg[:, :, (2, 1, 0)]

        # run detection
        xmin, ymin, xmax, ymax = detector.detect_and_visualize(image_list, imgg, args.dir, args.extension,
                                                               class_names, args.thresh, args.show_timer)
        img_out = imgg[(ymin) * 2:(ymax) * 2, (xmin) * 2:(xmax) * 2]
        # cv2.imshow('fig', img_out)
        # cv2.waitKey()

        center = ((xmin + xmax), (ymin + ymax))
        rot1, rot2 = rot.rotate(img_out, imgg, center)

        # rot1, rot2 = rot.rotate(img_out, img_out)
        # cv2.imshow('rot1',rot1)
        # cv2.imshow('rot2',rot2)
        # cv2.waitKey()

        (h, w) = rot1.shape[:2]
        # center = ((xmin+xmax), (ymin+ymax))
        # center = ((ymin+ymax),(xmin+xmax))
        h_new = int(sqrt(h ** 2 + w ** 2))
        w_new = h_new
        hmin = 10000
        M = cv2.getRotationMatrix2D(center, 0, 1.0)
        rotated1 = cv2.warpAffine(rot1, M, (w_new, h_new), borderValue=(0))
        rotated2 = cv2.warpAffine(rot2, M, (w_new, h_new), borderValue=(0))
        # cv2.imshow('rot1', rotated1)
        # cv2.imshow('rot2', rotated2)
        # cv2.waitKey()
        rotated1_new = np.zeros(imgg.shape)
        rotated2_new = np.zeros(imgg.shape)

        rotated1_new[max(512 - center[1], 0):min(512 + w_new - center[1], 1024),
        max(512 - center[0], 0):min(512 + w_new - center[0], 1024), :] = \
            rotated1[max(center[1] - 512, 0):min(center[1] + 512, h_new),
            max(center[0] - 512, 0):min(center[0] + 512, w_new), :]
        rotated2_new[max(512 - center[1], 0):min(512 + w_new - center[1], 1024),
        max(512 - center[0], 0):min(512 + w_new - center[0], 1024), :] = \
            rotated2[max(center[1] - 512, 0):min(center[1] + 512, h_new),
            max(center[0] - 512, 0):min(center[0] + 512, w_new), :]
        # cv2.imshow('rot1',rotated1_new)
        # cv2.imshow('rot2', rotated2_new)
        # cv2.waitKey()
        # rotated1_new = cv2.imread('./data/VOCdevkit/VOC2007/test/000001.jpg')
        cv2.imwrite('./data/VOCdevkit/VOC2007/test/000009_1.jpg', rotated1_new)
        xmin1, ymin1, xmax1, ymax1 = detector.detect_and_visualize('./data/VOCdevkit/VOC2007/test/000009_1.jpg',
                                                                   rotated1_new, args.dir, args.extension,
                                                                   class_names, args.thresh, args.show_timer)
        img_out1 = rotated1_new[(ymin1) * 2:(ymax1) * 2, (xmin1) * 2:(xmax1) * 2]
        cv2.imwrite('./data/VOCdevkit/VOC2007/test/000009_2.jpg', rotated2_new)
        xmin2, ymin2, xmax2, ymax2 = detector.detect_and_visualize('./data/VOCdevkit/VOC2007/test/000009_2.jpg',
                                                                   rotated2_new, args.dir, args.extension,
                                                                   class_names, args.thresh, args.show_timer)
        img_out2 = rotated2_new[(ymin2) * 2:(ymax2) * 2, (xmin2) * 2:(xmax2) * 2]

        if img_out.shape[0] < img_out2.shape[0]:
            rotated = rotated1_new
        else:
            rotated = rotated2_new

        hmin = 10000
        # rotated = []

        for i in np.arange(-8, 8):
            # rot_out = []
            center = (rotated.shape[1] // 2, rotated.shape[0] // 2)
            # print(rotated.shape)
            # print(center)

            M = cv2.getRotationMatrix2D(center, -i, 1.0)
            rotated_new = np.zeros(imgg.shape)
            rotated1 = cv2.warpAffine(rotated, M, (w_new, h_new), borderValue=(0))
            rotated_new[max(512 - center[1], 0):min(512 + w_new - center[1], 1024),
            max(512 - center[0], 0):min(512 + w_new - center[0], 1024), :] = \
                rotated1[max(center[1] - 512, 0):min(center[1] + 512, h_new),
                max(center[0] - 512, 0):min(center[0] + 512, w_new), :]

            cv2.imwrite('./data/VOCdevkit/VOC2007/test/000009_3.jpg', rotated_new)
            xmin, ymin, xmax, ymax = detector.detect_and_visualize('./data/VOCdevkit/VOC2007/test/000009_3.jpg',
                                                                   rotated_new, args.dir, args.extension,
                                                                   class_names, args.thresh, args.show_timer)
            rotated_new2 = rotated_new[(ymin) * 2:(ymax) * 2, (xmin - 10) * 2:(xmax + 10) * 2]

            habs = ymax - ymin
            if habs == 0:
                habs = 10000
            # print(habs)
            if habs < hmin:
                # print('okokokokokokok!!!!!!!!!!!!!!!!!')
                hmin = habs
                rot_out = rotated_new2.copy()

            # cv2.imshow('rot1',rotated1)
            # cv2.imshow('rot2', rotated2)
            # cv2.waitKey()
            # img_0 = np.zeros(imgg.shape)
            # img_0 =

            # xmin, ymin, xmax, ymax = detector.detect_and_visualize('1', rotated1, args.dir, args.extension,
            #                                                        class_names, args.thresh, args.show_timer)
            # img_out1 = imgg[(ymin) * 2:(ymax) * 2, (xmin) * 2:(xmax) * 2]
            #
            # xmin, ymin, xmax, ymax = detector.detect_and_visualize('2', rotated2, args.dir, args.extension,
            #                                                        class_names, args.thresh, args.show_timer)
            # img_out2 = imgg[(ymin) * 2:(ymax) * 2, (xmin) * 2:(xmax) * 2]
            #
            # if img_out1.shape[1] < img_out2.shape[1]:
            #     if img_out1.shape[1] < hmin:
            #         hmin = img_out1.shape[1]
            #         rotated = img_out1
            # else:
            #     if img_out2.shape[1] < hmin:
            #         hmin = img_out2.shape[1]
            #         rotated = img_out2

        # print(rot_out.shape)
        rot_out = np.uint8(rot_out)
        src = "/home/ad/dataset/outzheng/"
        if not os.path.exists(src):
            os.makedirs(src)

        cv2.imwrite(src + "{}.jpg".format(bann), rot_out,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])




        # cv2.imshow('rotate', rot_out)



        # img_0 = np.zeros(imgg.shape)
        # img_0[(ymin-2)*2:(ymax+2)*2, (xmin-2)*2:(xmax+2)*2] = imgg[(ymin-2)*2:(ymax+2)*2, (xmin-2)*2:(xmax+2)*2]
        # print(xmin, ymin, xmax, ymax)
        # img_0 = np.zeros(imgg.shape)
