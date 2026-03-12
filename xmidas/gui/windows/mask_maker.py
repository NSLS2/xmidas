import argparse
import logging
import sys
import webbrowser
import traceback
import os
import json
import scipy.stats as stats
import numpy as np
import pandas as pd
import tifffile as tf
import pyqtgraph as pg
import pyqtgraph.exporters
from glob import glob
from pyqtgraph import plot
from itertools import combinations
from scipy.stats import linregress
from packaging import version


from PyQt6 import QtWidgets, QtCore, QtGui, uic, QtTest
from PyQt6.QtGui import QMovie
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QApplication
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool, PYQT_VERSION_STR

from xmidas.utils import *
from xmidas.utils.color_maps import *
from xmidas.models.encoders import jsonEncoder
cmap_dict = create_color_maps()
ui_dir = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../layout"
))



class MaskSpecViewer(QtWidgets.QMainWindow):
    mask_signal: pyqtSignal = QtCore.pyqtSignal(np.ndarray)
    def __init__(self, xanes_stack=None, mask_map=None, energy=[], refs = None):
        super(MaskSpecViewer, self).__init__()
        uic.loadUi(os.path.join(ui_dir, "MaskSpecViewer.ui"), self)
        self.user_wd = os.path.abspath("~")

        self.xanes_stack = xanes_stack
        self.mask_map = mask_map
        self.energy = energy
        #self.mask_map = self.xanes_stack[-1]
        self.refs = refs #TODO fitting option 
        self.view_data()

        # connections
        self.dsb_low_threshold.valueChanged.connect(self.create_mask)
        self.dsb_high_threshold.valueChanged.connect(self.create_mask)
        self.pb_apply_mask.clicked.connect(self.apply_mask_to_xanes)
        self.action_export_mask.triggered.connect(self.export_mask)
        self.action_import_mask.triggered.connect(self.import_a_mask)
        self.actionLoad_Energy_List.triggered.connect(self.load_energy)
        self.actionLoad_XANES_Stack.triggered.connect(self.load_xanes_stack)
        self.actionLoad_XRF_Map.triggered.connect(lambda:self.load_mask_map())
        self.pb_send_mask.clicked.connect(self.send_mask)
        self.sldr_mask_low.valueChanged.connect(lambda value: self.dsb_low_threshold.setValue(value / 100.0))
        self.sldr_mask_high.valueChanged.connect(lambda value: self.dsb_high_threshold.setValue(value / 100.0))

    def view_data(self):

        self.xanes_view.setImage(self.xanes_stack)
        self.xanes_view.ui.menuBtn.hide()
        self.xanes_view.ui.roiBtn.hide()
        (self.dim1, self.dim3, self.dim2) = self.xanes_stack.shape
        self.xanes_view.setPredefinedGradient("viridis")
        self.xanes_view.setCurrentIndex(self.dim1 // 2)
        self.statusbar.showMessage("One image from the XANES stack is used as mask")
        self.xrf_view.setImage(self.mask_map)
        self.xrf_view.ui.menuBtn.hide()
        self.xrf_view.ui.roiBtn.hide()
        self.xrf_view.setPredefinedGradient("viridis")

        self.mask_view.ui.menuBtn.hide()
        self.mask_view.ui.roiBtn.hide()

    def create_mask(self):
        
        self.threshold_low = self.dsb_low_threshold.value()
        self.threshold_high = self.dsb_high_threshold.value()
        self.dsb_low_threshold.setMaximum(self.threshold_high-0.001)
        self.dsb_high_threshold.setMinimum(self.threshold_low+0.001)

        self.norm_mask = remove_nan_inf(self.mask_map) / np.nanmax(self.mask_map)
        self.norm_mask[(self.norm_mask < self.threshold_low) | 
                       (self.norm_mask > self.threshold_high)] = 0

        # self.norm_mask[self.norm_mask < self.threshold_low] = 0
        # self.norm_mask[self.norm_mask > self.threshold_high] = 0
        self.xrf_view.setImage(self.norm_mask)
        self.statusbar.showMessage("New Threshold Applied")
        self.binary_mask = np.where(self.norm_mask > 0, 1, 0)
        self.mask_view.setImage(self.binary_mask)
        

    def load_xanes_stack(self):
        """loading a new xanes stack"""
        filename = QFileDialog().getOpenFileName(self, "Select image data", "", "image file(*tiff *tif )")
        self.file_name = str(filename[0])
        self.xanes_stack = tf.imread(self.file_name).transpose(0, 2, 1)
        self.view_data()

    def load_energy(self):
        """To load energy list that will be used for plotting the spectra.
        number of stack should match length of energy list"""

        file_name = QFileDialog().getOpenFileName(self, "Open energy list", "", "text file (*.txt)")

        try:
            self.energy = np.loadtxt(file_name[0])
            logger.info("Energy file loaded")
            assert len(self.energy) == self.dim1
            self.view_data()

        except OSError:
            logger.error("No File selected")
            pass

    def load_mask_map(self, z = -1):
        """Array for masking.Z will be used if 3D """

        filename = QFileDialog().getOpenFileName(self, "Select image data", "", "image file(*tiff *tif )")
        self.xrf_file_name = str(filename[0])
        self.mask_map = tf.imread(self.xrf_file_name)
        if self.mask_map.ndim == 3:
            self.mask_map = self.mask_map[z]

        else:
            self.mask_map = self.mask_map.T

        assert (
            self.dim3,
            self.dim2,
        ) == self.mask_map.shape, f"Unexpected image dimensions: {self.mask_map.shape} vs {(self.dim2,self.dim3)}"

        self.view_data()
        self.create_mask()

    def send_mask(self):

        """ Apply mask to xanes viewer"""

        if self.cb_use_binary_mask.isChecked():
            self.mask_signal.emit(self.binary_mask)
        else:
            self.mask_signal.emit(self.norm_mask)

    def apply_mask_to_xanes(self):

        """Generates a mask with 0 and 1 from the choosen threshold and multply with the xanes stack.
        A spectrum will be generated from the new masked stack"""
        if self.cb_use_binary_mask.isChecked():
            self.masked_xanes = self.xanes_stack * self.binary_mask[np.newaxis,:,:]

        else:
            self.masked_xanes = self.xanes_stack * self.norm_mask[np.newaxis,:,:]

        self.xanes_view.setImage(self.masked_xanes)
        self.xanes_view.setCurrentIndex(self.dim1 // 2)
        self.statusbar.showMessage("Mask Applied to XANES")
        self.mask_spec = get_mean_spectra(self.masked_xanes)

        if len(self.energy) != 0:
            self.xdata = self.energy
        else:
            self.xdata = np.arange(0, self.dim1)
            self.statusbar.showMessage("No Energy List Available; Integer values are used for plotting")

        self.spectrum_view.plot(self.xdata, self.mask_spec, clear=True)

    def import_a_mask(self):
        filename = QFileDialog().getOpenFileName(self, "Select image data", "", "image file(*tiff *tif )")
        xrf_file_name = str(filename[0])
        self.binary_mask = tf.imread(xrf_file_name).T
        self.statusbar.showMessage("A New Mask Imported")
        self.mask_view.setImage(self.binary_mask)
        self.apply_mask_to_xanes()

    def export_mask(self):
        try:
            file_name = QFileDialog().getSaveFileName(self, "Save image data", "", "image file(*tiff *tif )")
            tf.imwrite(file_name[0] + ".tiff", self.binary_mask.T)
            logger.info(f"Updated Image Saved: {file_name[0]}")
            self.statusbar.showMessage("Mask Exported")
        except Exception:
            logger.error("No file to save")
            pass


class MaskMaker(QtWidgets.QMainWindow):
    mask_signal: pyqtSignal = QtCore.pyqtSignal(np.ndarray)
    def __init__(self, im_stack = None, mask_map=None):
        super(MaskMaker, self).__init__()
        uic.loadUi(os.path.join(ui_dir, "MaskMaker.ui"), self)
        self.user_wd = os.path.abspath("~")

        self.im_stack = im_stack
        self.mask_map = mask_map
        self.view_data()

        # connections
        self.dsb_low_threshold.valueChanged.connect(self.create_mask)
        self.dsb_high_threshold.valueChanged.connect(self.create_mask)
        self.action_export_mask.triggered.connect(self.export_mask)
        self.action_import_mask.triggered.connect(self.import_a_mask)
        self.actionLoad_XRF_Map.triggered.connect(lambda:self.load_mask_map())
        self.pb_send_mask.clicked.connect(self.send_mask)
        self.sldr_mask_low.valueChanged.connect(lambda value: self.dsb_low_threshold.setValue(value / 100.0))
        self.sldr_mask_high.valueChanged.connect(lambda value: self.dsb_high_threshold.setValue(value / 100.0))

    def view_data(self):
        self.xrf_view.setImage(self.mask_map)
        self.xrf_view.ui.menuBtn.hide()
        self.xrf_view.ui.roiBtn.hide()
        self.xrf_view.setPredefinedGradient("viridis")

        self.mask_view.ui.menuBtn.hide()
        self.mask_view.ui.roiBtn.hide()

    def create_mask(self):
        
        self.threshold_low = self.dsb_low_threshold.value()
        self.threshold_high = self.dsb_high_threshold.value()
        self.dsb_low_threshold.setMaximum(self.threshold_high-0.001)
        self.dsb_high_threshold.setMinimum(self.threshold_low+0.001)

        self.norm_mask = remove_nan_inf(self.mask_map) / np.nanmax(self.mask_map)
        self.norm_mask[(self.norm_mask < self.threshold_low) | 
                       (self.norm_mask > self.threshold_high)] = 0

        # self.norm_mask[self.norm_mask < self.threshold_low] = 0
        # self.norm_mask[self.norm_mask > self.threshold_high] = 0
        self.xrf_view.setImage(self.norm_mask)
        self.statusbar.showMessage("New Threshold Applied")
        self.binary_mask = np.where(self.norm_mask > 0, 1, 0)
        self.mask_view.setImage(self.binary_mask)


    def load_mask_map(self, z = -1):
        """Array for masking.Z will be used if 3D """

        filename = QFileDialog().getOpenFileName(self, "Select image data", "", "image file(*tiff *tif )")
        self.xrf_file_name = str(filename[0])
        self.mask_map = tf.imread(self.xrf_file_name)
        if self.mask_map.ndim == 3:
            self.mask_map = self.mask_map[z]

        else:
            self.mask_map = self.mask_map.T

        assert (
            self.dim3,
            self.dim2,
        ) == self.mask_map.shape, f"Unexpected image dimensions: {self.mask_map.shape} vs {(self.dim2,self.dim3)}"

        self.view_data()
        self.create_mask()

    def send_mask(self):

        """ Apply mask to xanes viewer"""

        if self.cb_use_binary_mask.isChecked():
            self.mask_signal.emit(self.im_stack*self.binary_mask[np.newaxis,:,:])
        else:
            self.mask_signal.emit(self.im_stack*self.norm_mask[np.newaxis,:,:])

    def import_a_mask(self):
        filename = QFileDialog().getOpenFileName(self, "Select image data", "", "image file(*tiff *tif )")
        xrf_file_name = str(filename[0])
        self.binary_mask = tf.imread(xrf_file_name).T
        self.statusbar.showMessage("A New Mask Imported")
        self.mask_view.setImage(self.binary_mask)
        self.apply_mask_to_xanes()

    def export_mask(self):
        try:
            file_name = QFileDialog().getSaveFileName(self, "Save image data", "", "image file(*tiff *tif )")
            tf.imwrite(file_name[0] + ".tiff", self.binary_mask.T)
            logger.info(f"Updated Image Saved: {file_name[0]}")
            self.statusbar.showMessage("Mask Exported")
        except Exception:
            logger.error("No file to save")
            pass
