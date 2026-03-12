import argparse
import logging
import datetime
import sys
import webbrowser
import traceback
import os
import json

import numpy as np
import pandas as pd
import tifffile as tf
import pyqtgraph as pg


from PyQt6 import QtWidgets, QtCore, QtGui, uic, QtTest
from PyQt6.QtGui import QMovie
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QApplication
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool, PYQT_VERSION_STR

from xmidas.utils import *
from xmidas.utils.color_maps import *

cmap_dict = create_color_maps()

#from . import __version__

logger = logging.getLogger()
try:
    import cv2  # noqa: F401
except Exception:
    logger.warning("openCV module not found")
    pass
if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

ui_dir = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../layout"
))


class singleStackViewer(QtWidgets.QMainWindow):
    def __init__(self, img_stack, gradient="viridis"):
        super(singleStackViewer, self).__init__()

        # Load the UI Page
        uic.loadUi(os.path.join(ui_dir, "singleStackView.ui"), self)
        self.user_wd = os.path.abspath("~")

        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()

        self.img_stack = img_stack
        self.gradient = gradient
        self.image_view.setPredefinedGradient(gradient)

        if self.img_stack.ndim == 3:
            self.dim1, self.dim3, self.dim2 = img_stack.shape
        elif self.img_stack.ndim == 2:
            self.dim3, self.dim2 = img_stack.shape
            self.dim1 = 1
        self.hs_img_stack.setMaximum(self.dim1 - 1)
        self.hs_img_stack.setValue(np.round(self.dim1 / 2))
        self.displayStack()

        # connections
        self.hs_img_stack.valueChanged.connect(self.displayStack)
        self.actionSave.triggered.connect(self.saveImageStackAsTIFF)

    def displayStack(self):
        im_index = self.hs_img_stack.value()
        if self.img_stack.ndim == 2:
            self.image_view.setImage(self.img_stack)
        else:
            self.image_view.setImage(self.img_stack[im_index])
        self.label_img_count.setText(f"{im_index + 1}/{self.dim1}")

    def saveImageStackAsTIFF(self):
        file_name = QFileDialog().getSaveFileName(self, 
                                                  "Export Stack", 
                                                  os.path.join(self.user_wd, "image_stack.tiff"), 
                                                  "*.tiff;;*.tif")
        if file_name[0]:
            if self.img_stack.ndim == 3:
                tf.imwrite(file_name[0], np.float32(self.img_stack.transpose(0, 2, 1)))
            elif self.img_stack.ndim == 2:
                tf.imwrite(file_name[0], np.float32(self.img_stack.T))

            self.user_wd = os.path.dirname(file_name[0])
        else:
            pass
