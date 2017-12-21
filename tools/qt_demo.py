#!/usr/bin/env python
# Copyright (c) 2008-14 Qtrac Ltd. All rights reserved.
# This program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 2 of the License, or
# version 3 of the License, or (at your option) any later version. It is
# provided for educational purposes and is distributed in the hope that
# it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
# the GNU General Public License for more details.

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import sys
from PyQt5.QtCore import (QTime, QTimer, Qt)
from PyQt5.QtWidgets import QApplication , QMainWindow, QLabel
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from number import *
import cv2

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

    def show_picture(self, img, img0):
        self.naive_img.setScaledContents(True)
        self.ssd_img.setScaledContents(True)
        self.naive_img.setPixmap(QPixmap.fromImage(img0))
        self.ssd_img.setPixmap(QPixmap.fromImage(img))





if __name__ == '__main__':

    bann = '000009'
    src = "/home/ad/dataset/outzheng5/{}.jpg".format(bann)
    src0 = "/home/ad/dataset/20171213/select/{}.jpg".format(bann)
    print(src)
    # src="./outzheng5/000009.jpg"
    A = cv2.imread(src)
    A0 = cv2.imread(src0)
    # print(A.shape)

    qimg = QImage(A.tostring(), A.shape[1], A.shape[0], QImage.Format_RGB888).rgbSwapped()
    qimg0 = QImage(A0.tostring(), A0.shape[1], A0.shape[0], QImage.Format_RGB888).rgbSwapped()

    # cv2.imshow('A', A)
    # cv2.waitKey()
    #
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    myWin.show_picture(qimg, qimg0)
    sys.exit(app.exec_())
