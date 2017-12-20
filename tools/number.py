# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'number.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(946, 681)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.navie_text = QtWidgets.QLabel(self.centralwidget)
        self.navie_text.setGeometry(QtCore.QRect(10, 50, 131, 21))
        self.navie_text.setObjectName("navie_text")
        self.naive_img = QtWidgets.QLabel(self.centralwidget)
        self.naive_img.setGeometry(QtCore.QRect(10, 80, 361, 331))
        self.naive_img.setObjectName("naive_img")
        self.ssd_text = QtWidgets.QLabel(self.centralwidget)
        self.ssd_text.setGeometry(QtCore.QRect(10, 450, 131, 21))
        self.ssd_text.setObjectName("ssd_text")
        self.ssd_img = QtWidgets.QLabel(self.centralwidget)
        self.ssd_img.setGeometry(QtCore.QRect(10, 510, 401, 101))
        self.ssd_img.setObjectName("ssd_img")
        self.line = QtWidgets.QLabel(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(590, 50, 101, 41))
        self.line.setObjectName("line")
        self.result_label = QtWidgets.QLabel(self.centralwidget)
        self.result_label.setGeometry(QtCore.QRect(610, 450, 54, 12))
        self.result_label.setObjectName("result_label")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(610, 470, 291, 41))
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(610, 520, 271, 41))
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(590, 130, 331, 141))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 946, 17))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.navie_text.setText(_translate("MainWindow", "navie picture"))
        self.naive_img.setText(_translate("MainWindow", "TextLabel"))
        self.ssd_text.setText(_translate("MainWindow", "SSD result"))
        self.ssd_img.setText(_translate("MainWindow", "TextLabel"))
        self.line.setText(_translate("MainWindow", "get line"))
        self.result_label.setText(_translate("MainWindow", "Result"))

