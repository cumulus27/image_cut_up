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
        self.gray = []
        self.grayline1 = []
        self.grayline2 = []


    def white_balance(self, h):
        pass


