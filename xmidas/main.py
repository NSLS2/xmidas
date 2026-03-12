# -*- coding: utf-8 -*-

# Author: Ajith Pattammattel
# First Version on:06-23-2020
#python 3.12 update on July 2025
__version__ = "1.0.0"

import argparse
import logging
import datetime
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
import faulthandler
faulthandler.enable()


from glob import glob
from pyqtgraph import plot
from itertools import combinations
from scipy.stats import linregress
from packaging import version

from PyQt6 import QtWidgets, QtCore, QtGui, uic, QtTest
from PyQt6.QtGui import QMovie
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QApplication
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool, PYQT_VERSION_STR, QTimer


from xmidas.utils.utils import *
from xmidas.utils.color_maps import create_color_maps
from xmidas.models.encoders import jsonEncoder

from xmidas.gui.windows.xanes_viewer import XANESViewer
from xmidas.gui.windows.multichannel_viewer import MultiChannelWindow
from xmidas.gui.windows.mask_maker import MaskSpecViewer
from xmidas.gui.windows.singleStackViewer import *

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
    "gui/layout"
))

print(ui_dir)

# global settings for pyqtgraph plot and image colormaps
pg.setConfigOption("imageAxisOrder", "row-major")

class midasWindow(QtWidgets.QMainWindow):
    def __init__(self, im_stack=None, energy=[], refs=[]):
        super(midasWindow, self).__init__()
        uic.loadUi(os.path.join(ui_dir, "midasMainwindow.ui"), self)
        self.im_stack = im_stack
        self.energy = energy
        self.refs = refs
        self.loaded_tranform_file = []
        self.image_roi2_flag = False
        self.refStackAvailable = False
        self.isAReload = False
        self.plotWidth = 2
        self.stackStatusDict = {}

        # Performance optimization: debounce timer and cache
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._do_update_stack)
        self._pending_update = False
        
        # Cache for expensive operations
        self._cache = {
            'resize': {},      # {scaling_factor: processed_stack}
            'outliers': {},    # {nsigma: processed_stack}
            'smooth': {},      # {window_size: processed_stack}
            'last_params': {}  # Track last used parameters
        }

        self.user_wd = os.path.expanduser("~")
        # self.user_config_path = os.path.join(ui_path,"user_config.json")
        
        # if not os.path.exists(self.user_config_path):
            
        #     with open(f"{self.user_config_path}", "w") as fp:
        #         json.dump(self.user_config, fp, indent=4)

        self.plt_colors = [
            "g",
            "r",
            "c",
            "m",
            "y",
            "w",
            "b",
            pg.mkPen(70, 5, 80),
            pg.mkPen(255, 85, 130),
            pg.mkPen(0, 85, 130),
            pg.mkPen(255, 170, 60),
        ] * 3
        # window style
        self.actionDarkMode.triggered.connect(self.darkMode)
        self.actionDefault.triggered.connect(self.defaultMode)
        self.actionModern.triggered.connect(self.modernMode)

        # self.setToolTipsVisible(True)
        for menuItem in self.findChildren(QtWidgets.QMenu):
            menuItem.setToolTipsVisible(True)

        # plotview options
        self.actionWhite.triggered.connect(lambda: self.spectrum_view.setBackground("w"))
        self.actionRed.triggered.connect(lambda: self.spectrum_view.setBackground("r"))
        self.actionYellow.triggered.connect(lambda: self.spectrum_view.setBackground("y"))
        self.actionBlue.triggered.connect(lambda: self.spectrum_view.setBackground("b"))
        self.actionBlack.triggered.connect(lambda: self.spectrum_view.setBackground((0, 0, 0)))

        self.actn1.triggered.connect(lambda: self.setPlotLineWidth(int(self.actn1.text())))
        self.actn2.triggered.connect(lambda: self.setPlotLineWidth(int(self.actn2.text())))
        self.actn3.triggered.connect(lambda: self.setPlotLineWidth(int(self.actn3.text())))
        self.actn4.triggered.connect(lambda: self.setPlotLineWidth(int(self.actn4.text())))
        self.actn5.triggered.connect(lambda: self.setPlotLineWidth(int(self.actn5.text())))
        self.actn6.triggered.connect(lambda: self.setPlotLineWidth(int(self.actn6.text())))
        self.actn8.triggered.connect(lambda: self.setPlotLineWidth(int(self.actn8.text())))
        self.actn10.triggered.connect(lambda: self.setPlotLineWidth(int(self.actn10.text())))

        self.actionOpen_Image_Data.triggered.connect(self.browse_file)
        self.actionOpen_Multiple_Files.triggered.connect(self.createVirtualStack)
        self.actionSave_as.triggered.connect(lambda: self.save_stack())
        self.actionExit.triggered.connect(lambda: QApplication.closeAllWindows())
        self.actionOpen_in_GitHub.triggered.connect(self.open_github_link)
        self.actionLoad_Energy.triggered.connect(self.select_elist)
        self.menuFile.setToolTipsVisible(True)

        # Accessories
        self.actionOpen_Mask_Gen.triggered.connect(self.openMaskMaker)
        self.actionMultiColor.triggered.connect(self.openMultiColorWindow)

        # calculations
        self.pb_transpose_stack.clicked.connect(lambda: self.threadMaker(self.transposeStack))
        self.pb_swapXY_stack.clicked.connect(lambda: self.threadMaker(self.swapStackXY))
        self.pb_reset_img.clicked.connect(self.reloadImageStack)
        self.pb_crop.clicked.connect(self.crop_to_dim)
        self.pb_apply_crop_to_all.clicked.connect(self.apply_crop_to_all)
        self.pb_crop.clicked.connect(self.view_stack)
        self.sb_scaling_factor.valueChanged.connect(self.view_stack)
        self.pb_ref_xanes.clicked.connect(self.select_ref_file)
        self.pb_elist_xanes.clicked.connect(self.select_elist)

        # batchjobs
        self.actionPlotAllCorrelations.triggered.connect(self.plotCorrelationsAllCombinations)

        # Use debounced updates for sliders to prevent rapid successive processing
        [
            uis.valueChanged.connect(self.schedule_update)
            for uis in [self.hs_smooth_size, self.hs_nsigma, self.hs_bg_threshold]
        ]

        [
            uis.stateChanged.connect(self.replot_image)
            for uis in [self.cb_remove_bg, self.cb_remove_outliers, self.cb_smooth, self.cb_norm, self.cb_log]
        ]

        [
            uis.stateChanged.connect(self.view_stack)
            for uis in [self.cb_remove_edges, self.cb_upscale, self.cb_rebin]
        ]

        # ToolBar
        self.actionStack_Info.triggered.connect(self.displayStackInfo)
        self.actionSave_Image.triggered.connect(self.save_disp_img)
        self.actionExport_Stack.triggered.connect(lambda: self.save_stack())

        # ROI background
        self.actionSubtract_ROI_BG.triggered.connect(lambda: self.threadMaker(self.removeROIBGStack))

        # alignment
        self.pb_load_align_ref.clicked.connect(self.loadAlignRefImage)
        self.pb_loadAlignTranform.clicked.connect(self.importAlignTransformation)
        self.pb_saveAlignTranform.clicked.connect(self.exportAlignTransformation)
        self.pb_alignStack.clicked.connect(lambda: self.threadMaker(self.stackRegistration))
        # self.pb_alignStack.clicked.connect(self.stackRegistration)

        # save_options
        self.actionSave_Sum_Image.triggered.connect(lambda: self.save_stack(method="sum"))
        self.actionSave_Mean_Image.triggered.connect(lambda: self.save_stack(method="mean"))
        self.actionExport_Image_to_CSV.triggered.connect(self.stackToCSV)
        self.pb_save_disp_spec.clicked.connect(self.save_disp_spec)
        self.actionSave_Energy_List.triggered.connect(self.saveEnergyList)
        self.pb_show_roi.clicked.connect(self.getROIMask)
        self.pb_addToCollector.clicked.connect(self.addSpectrumToCollector)
        self.pb_collect_clear.clicked.connect(lambda: self.spectrum_view_collect.clear())
        self.pb_saveCollectorPlot.clicked.connect(self.saveCollectorPlot)

        # XANES Normalization
        self.pb_apply_xanes_norm.clicked.connect(self.nomalizeLiveSpec)
        self.pb_auto_Eo.clicked.connect(self.findEo)
        self.pb_xanes_norm_vals.clicked.connect(self.initNormVals)
        self.pb_apply_norm_to_stack.clicked.connect(lambda: self.threadMaker(self.normalizeStack))
        self.actionExport_Norm_Params.triggered.connect(self.exportNormParams)
        self.actionImport_Norm_Params.triggered.connect(self.importNormParams)

        # Analysis
        self.pb_pca_scree.clicked.connect(self.pca_scree_)
        self.pb_calc_components.clicked.connect(self.calc_comp_)
        self.pb_kmeans_elbow.clicked.connect(self.kmeans_elbow)
        self.pb_calc_cluster.clicked.connect(self.clustering_)
        self.pb_xanes_fit.clicked.connect(self.fast_xanes_fitting)
        self.pb_plot_refs.clicked.connect(self.plt_xanes_refs)

        self.show()

        self.threadpool = QThreadPool()
        print(f"Multithreading with maximum  {self.threadpool.maxThreadCount()} threads")

    # View Options
    def darkMode(self):
        self.centralwidget.setStyleSheet(open(os.path.join(ui_dir, "css/darkStyle.css")).read())

    def defaultMode(self):
        self.centralwidget.setStyleSheet(open(os.path.join(ui_dir, "css/defaultStyle.css")).read())

    def modernMode(self):
        self.centralwidget.setStyleSheet(open(os.path.join(ui_dir, "css/modern.css")).read())

    def setPlotLineWidth(self, width_input):
        self.plotWidth = width_input
        try:
            self.update_spectrum()
        except Exception:
            pass

    def openMultiColorWindow(self):
        self.multicolorwindow = MultiChannelWindow()
        self.multicolorwindow.show()

    def openMaskMaker(self):
        self.mask_window = MaskSpecViewer(xanes_stack=self.displayedStack, energy=self.energy)
        self.mask_window.show()

    def open_github_link(self):
        webbrowser.open("https://github.com/pattammattel/NSLS-II-MIDAS/wiki")

    def threadMaker(self, funct):
        # Pass the function to execute
        worker = Worker(funct)  # Any other args, kwargs are passed to the run function
        self.loadSplashScreen()
        worker.signals.start.connect(self.splash.startAnimation)
        worker.signals.result.connect(self.print_output)

        list(
            map(
                worker.signals.finished.connect,
                [
                    self.thread_complete,
                    self.splash.stopAnimation,
                    self.update_stack_info,
                    self.update_spectrum,
                    self.update_image_roi,
                    self.setImageROI
                ],
            )
        )

        # Execute
        self.threadpool.start(worker)


    # File Loading

    def createVirtualStack(self):
        """User can load multiple/series of tiff images with same shape.
        The 'self.load_stack()' recognizes 'self.filename as list and create the stack.
        """
        self.energy = []
        filter = "TIFF (*.tiff);;TIF (*.tif);;all_files (*)"
        file_name = QFileDialog()
        file_name.setFileMode(QFileDialog.FileMode.ExistingFiles)
        names = file_name.getOpenFileNames(self, "Open files", self.user_wd, filter)
        if names[0]:

            self.file_name = names[0]
            self.user_wd = os.path.dirname(self.file_name[0])
            self.load_stack()

        else:
            self.statusbar_main.showMessage("No file has selected")
            pass

    def load_stack(self):

        """load the image data from the selected file.
        If the the choice is for multiple files stack will be created in a loop.
        If single h5 file is selected the unpacking will be done with 'get_xrf_data' function in StackCalcs.
        From the h5 the program can recognize the beamline. The exported stack will be normalized to I0.

        If the single tiff file is choosen tf.imread() is used.

        The output 'self.im_stack' is the unmodified data file
        """

        # Invalidate cache when loading new data
        self.invalidate_cache()
        
        self.log_warning = False  # for the Qmessage box in cb_log
        self.image_roi2_flag = False
        self.cb_log.setChecked(False)
        self.cb_remove_edges.setChecked(False)
        self.cb_norm.setChecked(False)
        self.cb_smooth.setChecked(False)
        self.cb_remove_outliers.setChecked(False)
        self.cb_remove_bg.setChecked(False)
        self.cb_rebin.setChecked(False)
        self.cb_upscale.setChecked(False)
        self.sb_xrange1.setValue(0)
        self.sb_yrange1.setValue(0)
        self.sb_zrange1.setValue(0)

        self.menuMask.setEnabled(True)
        self.actionLoad_Energy.setEnabled(True)
        self.actionSave_Energy_List.setEnabled(True)
        self.actionSave_as.setEnabled(True)

        self.sb_zrange2.setMaximum(99999)
        self.sb_xrange2.setMaximum(99999)
        self.sb_yrange2.setMaximum(99999)

        self.statusbar_main.showMessage("Loading.. please wait...")

        if isinstance(self.file_name, list):

            all_images = []

            for im_file in self.file_name:
                img = tf.imread(im_file)
                all_images.append(img)  # row major image
            self.im_stack = np.dstack(all_images).transpose((2, 0, 1))
            self.avgIo = 1  # I0 is only applicable to XRF h5 files
            self.sb_zrange2.setValue(self.im_stack.shape[0])

        else:

            if self.file_name.endswith(".h5"):
                self.im_stack, mono_e, bl_name, self.avgIo = get_xrf_data(self.file_name)
                self.statusbar_main.showMessage(f"Data from {bl_name}")
                self.sb_zrange2.setValue(mono_e / 10)
                self.energy = []

            elif self.file_name.endswith(".tiff") or self.file_name.endswith(".tif"):
                self.im_stack_ = tf.imread(self.file_name)
                if self.im_stack_.ndim == 2:
                    self.im_stack = self.im_stack_.reshape(1, self.im_stack_.shape[0], self.im_stack_.shape[1])

                else:
                    self.im_stack = self.im_stack_
                self.sb_zrange2.setValue(self.im_stack.shape[0])
                self.autoEnergyLoader()
                self.energyUnitCheck()
                self.avgIo = 1

            else:
                logger.error("Unknown data format")

        """ Fill the stack dimensions to the GUI and set the image dimensions as max values.
         This prevent user from choosing higher image dimensions during a resizing event"""

        logger.info(f" loaded stack with {np.shape(self.im_stack)} from the file")

        try:
            logger.info(f" Transposed to shape: {np.shape(self.im_stack)}")
            self.init_dimZ, self.init_dimY, self.init_dimX = self.im_stack.shape
            # Remove any previously set max value during a reload

            self.sb_xrange2.setValue(self.init_dimX)
            self.sb_yrange2.setValue(self.init_dimY)

        except UnboundLocalError:
            logger.error("No file selected")
            pass

        self.view_stack()
        logger.info("Stack displayed correctly")
        self.update_stack_info()

        logger.info(f"completed image shape {np.shape(self.im_stack)}")

        try:
            self.statusbar_main.showMessage(f"Loaded: {self.file_name}")

        except AttributeError:
            self.statusbar_main.showMessage("New Stack is made from selected tiffs")
            pass

    def browse_file(self):
        """To open a file widow and choose the data file.
        The filename will be used to load data using 'rest and load stack' function"""

        filename = QFileDialog().getOpenFileName(
            self, "Select image data", self.user_wd, "image file(*.hdf *.h5 *tiff *tif )"
        )
        self.file_name = str(filename[0])
        self.user_wd = os.path.dirname(self.file_name)

        # if user decides to cancel the file window gui returns to original state
        if self.file_name:
            self.disconnectImageActions()
            self.isAReload = False
            self.load_stack()

        else:
            self.statusbar_main.showMessage("No file has selected")
            pass

    def autoEnergyLoader(self):

        dir_, filename_ = os.path.split(self.file_name)
        self.efilePath_name = os.path.join(dir_, os.path.splitext(filename_)[0] + ".txt")
        self.efilePath_log = os.path.join(dir_, "maps_log_tiff.txt")

        if os.path.isfile(self.efilePath_name):
            self.efilePath = self.efilePath_name
            self.efileLoader()
            self.statusbar_main.showMessage(f"Energy File detected {self.efilePath}")

        elif os.path.isfile(self.efilePath_log):
            self.efilePath = self.efilePath_log
            self.efileLoader()
            self.statusbar_main.showMessage(f"Energy File detected {self.efilePath}")

        else:
            self.efilePath = False
            self.efileLoader()

    def update_stack_info(self):
        z, y, x = np.shape(self.displayedStack)
        self.sb_zrange2.setMaximum(z + self.sb_zrange1.value())
        self.sb_xrange2.setValue(x)
        self.sb_xrange2.setMaximum(x)
        self.sb_yrange2.setValue(y)
        self.sb_yrange2.setMaximum(y)
        logger.info("Stack info has been updated")

    # Image Transformations

    def crop_to_dim(self):
        self.x1, self.x2 = self.sb_xrange1.value(), self.sb_xrange2.value()
        self.y1, self.y2 = self.sb_yrange1.value(), self.sb_yrange2.value()
        self.z1, self.z2 = self.sb_zrange1.value(), self.sb_zrange2.value()

        try:
            self.displayedStack = remove_nan_inf(
                self.displayedStack[self.z1 : self.z2, self.y1 : self.y2, self.x1 : self.x2]
            )
        except Exception:
            self.displayedStack = remove_nan_inf(
                self.im_stack[self.z1 : self.z2, self.y1 : self.y2, self.x1 : self.x2]
            )

    def apply_crop_to_all(self):
        dir_ = os.path.dirname(self.file_name)
        tiffs = glob(dir_+"/*.tiff")

        self.x1, self.x2 = self.sb_xrange1.value(), self.sb_xrange2.value()
        self.y1, self.y2 = self.sb_yrange1.value(), self.sb_yrange2.value()
        self.z1, self.z2 = self.sb_zrange1.value(), self.sb_zrange2.value()

        #print(tiffs)
        save_str = ' '
        for fname in tiffs:
            print(fname)
            im_array = tf.imread(fname)
            im_name = os.path.join(dir_,os.path.basename(fname).split('.')[0]+"_cropped.tiff")
            save_path = os.path.relpath(im_name, os.path.expanduser('~'))
            if np.ndim(im_array) == 3:
                tf.imwrite(im_name, im_array[self.z1 : self.z2, self.y1 : self.y2, self.x1 : self.x2])
                save_str+=f"\n{save_path} cropped from {im_array.shape} to (z,y,x):{self.z1}:{self.z2},{self.y1}:{self.y2},{self.x1}:{self.x2}"
                logger.info(f"{save_path} saved")
            elif np.ndim(im_array) == 2:
                tf.imwrite(im_name, im_array[self.y1 : self.y2, self.x1 : self.x2])
                logger.info(f"{save_path} saved")
                save_str+=f"\n{save_path} cropped from {im_array.shape} to (z,y,x):{self.y1}:{self.y2},{self.x1}:{self.x2}"
            else:
                pass

            print(f"{save_str = }")

        # Get the current date and time
        current_datetime = datetime.datetime.now()
        formatted_date_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        # save crop settings
        file_name = os.path.join(dir_, f"crop_log_{formatted_date_time}.txt")

        with open(file_name, 'w') as file:
            file.write(save_str)

        #print(f"log saved as '{file_name}'")

    def transpose_stack(self):
        self.displayedStack = self.displayedStack.T
        self.update_spectrum()
        self.update_spec_image_roi()

    # Alignement

    def loadAlignRefImage(self):
        filename = QFileDialog().getOpenFileName(self, "Image Data", self.user_wd, "*.tiff *.tif")
        file_name = str(filename[0])
        self.user_wd = os.path.dirname(file_name)
        self.alignRefImage = tf.imread(file_name)
        assert self.alignRefImage.shape == self.displayedStack.shape, "Image dimensions do not match"
        self.refStackAvailable = True
        self.rb_alignRefVoid.setChecked(False)
        self.change_color_on_load(self.pb_load_align_ref)

    def stackRegistration(self):

        self.transformations = {
            "TRANSLATION": StackReg.TRANSLATION,
            "RIGID_BODY": StackReg.RIGID_BODY,
            "SCALED_ROTATION": StackReg.SCALED_ROTATION,
            "AFFINE": StackReg.AFFINE,
            "BILINEAR": StackReg.BILINEAR,
        }

        self.transformType = self.transformations[self.cb_alignTransform.currentText()]
        self.alignReferenceImage = self.cb_alignRef.currentText()
        self.alignRefStackVoid = self.rb_alignRefVoid.isChecked()
        self.alignMaxIter = self.sb_maxIterVal.value()

        if self.cb_use_tmatFile.isChecked():

            if len(self.loaded_tranform_file) > 0:

                self.displayedStack = align_with_tmat(
                    self.displayedStack, tmat_file=self.loaded_tranform_file, transformation=self.transformType
                )
                logger.info("Aligned to the tranform File")

            else:
                logger.error("No Tranformation File Loaded")

        elif self.cb_iterAlign.isChecked():

            if not self.refStackAvailable:
                self.alignRefImage = self.displayedStack
            else:
                pass

            self.displayedStack = align_stack_iter(
                self.displayedStack,
                ref_stack_void=False,
                ref_stack=self.alignRefImage,
                transformation=self.transformType,
                method=("previous", "first"),
                max_iter=self.alignMaxIter,
            )

        else:
            if not self.refStackAvailable:
                self.alignRefImage = self.displayedStack

            else:
                pass

            self.displayedStack, self.tranform_file = align_stack(
                self.displayedStack,
                ref_image_void=True,
                ref_stack=self.alignRefImage,
                transformation=self.transformType,
                reference=self.alignReferenceImage,
            )
            logger.info("New Tranformation file available")
        self.im_stack = self.displayedStack

    def exportAlignTransformation(self):

        

        file_name = QFileDialog().getSaveFileName(
                                        self, 
                                        "Save Transformation File", 
                                        os.path.join(self.user_wd,"TranformationMatrix.npy"), 
                                        "text file (*.npy)"
                                        )
        if file_name[0]:
            np.save(file_name[0], self.tranform_file)
            self.user_wd = os.path.dirname(file_name[0])
        else:
            pass

    def importAlignTransformation(self):
        file_name = QFileDialog().getOpenFileName(self, "Open Transformation File", self.user_wd, "text file (*.npy)")
        if file_name[0]:
            self.loaded_tranform_file = np.load(file_name[0])
            self.cb_use_tmatFile.setChecked(True)
            self.user_wd = os.path.dirname(file_name[0])
            logger.info("Transformation File Loaded")
        else:
            pass

    def loadSplashScreen(self):
        self.splash = LoadingScreen()

        px = self.geometry().x()
        py = self.geometry().y()
        ph = self.geometry().height()
        pw = self.geometry().width()
        dw = self.splash.width()
        dh = self.splash.height()
        new_x, new_y = px + (0.5 * pw) - dw, py + (0.5 * ph) - dh
        self.splash.setGeometry(int(new_x), int(new_y), int(dw), int(dh))

        self.splash.show()

    def reloadImageStack(self):
        self.isAReload = True
        self.load_stack()

    def schedule_update(self):
        """Schedule a debounced update to prevent rapid successive processing."""
        self.update_timer.stop()
        self.update_timer.start(300)  # 300ms delay
        
    def _do_update_stack(self):
        """Internal method called by timer to perform the actual update."""
        self.replot_image()
        
    def invalidate_cache(self):
        """Clear all cached results when base stack changes."""
        self._cache = {
            'resize': {},
            'outliers': {},
            'smooth': {},
            'last_params': {}
        }
        logger.info("Cache invalidated")

    def update_stack(self):
        """Update the displayed stack with all selected modifications.
        
        Uses caching to avoid redundant processing of expensive operations.
        
        OPERATION ORDER (FIXED):
        The operations are applied in a specific order optimized for:
        1. Performance (reduce data size early)
        2. Correctness (irreversible operations like log/edge removal in proper sequence)
        
        Order:
        1. Resize/Rebin - Reduces data size for faster subsequent operations
        2. Remove Edges - Reduces data size, IRREVERSIBLE
        3. Remove Outliers - Pixel-level operation
        4. Remove Background - Threshold-based filtering
        5. Log Transform - IRREVERSIBLE, changes data scale
        6. Smoothing - Works on any scale
        7. Normalization - Final scaling step
        
        Note: If you need a different order, you'll need to apply operations
        manually in sequence rather than using checkboxes simultaneously.
        """
        # Show busy cursor during processing
        QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        
        try:
            self.displayedStack = self.im_stack
            self.crop_to_dim()

            # OPTIMIZATION 1: Resize/rebin first to reduce data size for subsequent operations
            if self.cb_rebin.isChecked():
                self.cb_upscale.setChecked(False)
                self.sb_scaling_factor.setEnabled(True)
                scaling_factor = self.sb_scaling_factor.value()
                
                # Check cache
                cache_key = f"rebin_{scaling_factor}"
                if cache_key in self._cache['resize']:
                    self.displayedStack = self._cache['resize'][cache_key]
                    logger.info(f"Using cached rebin result (factor={scaling_factor})")
                else:
                    self.displayedStack = resize_stack(self.displayedStack, scaling_factor=scaling_factor)
                    self._cache['resize'][cache_key] = self.displayedStack.copy()
                self.update_stack_info()

            elif self.cb_upscale.isChecked():
                self.cb_rebin.setChecked(False)
                self.sb_scaling_factor.setEnabled(True)
                scaling_factor = self.sb_scaling_factor.value()
                
                # Check cache
                cache_key = f"upscale_{scaling_factor}"
                if cache_key in self._cache['resize']:
                    self.displayedStack = self._cache['resize'][cache_key]
                    logger.info(f"Using cached upscale result (factor={scaling_factor})")
                else:
                    self.displayedStack = resize_stack(
                        self.displayedStack, upscaling=True, scaling_factor=scaling_factor
                    )
                    self._cache['resize'][cache_key] = self.displayedStack.copy()
                self.update_stack_info()

            # OPTIMIZATION 2: Remove edges early to reduce data size
            if self.cb_remove_edges.isChecked():
                self.displayedStack = remove_edges(self.displayedStack)
                logger.info(f"Removed edges, new shape {self.displayedStack.shape}")
                self.update_stack_info()

            # Outlier removal with caching
            if self.cb_remove_outliers.isChecked():
                self.hs_nsigma.setEnabled(True)
                nsigma = self.hs_nsigma.value() / 10
                
                # Only cache if this is the ONLY operation (no subsequent ops that would be skipped)
                # Otherwise the cache would bypass operations like log transform
                self.displayedStack = remove_hot_pixels(self.displayedStack, NSigma=nsigma)
                    
                self.label_nsigma.setText(str(nsigma))
                logger.info(f"Removing Outliers with NSigma {nsigma}")
            else:
                self.hs_nsigma.setEnabled(False)

            # Background removal
            if self.cb_remove_bg.isChecked():
                self.hs_bg_threshold.setEnabled(True)
                logger.info("Removing background")
                bg_threshold = self.hs_bg_threshold.value()
                self.label_bg_threshold.setText(str(bg_threshold) + "%")
                self.displayedStack = clean_stack(self.displayedStack, auto_bg=False, bg_percentage=bg_threshold)
            else:
                self.hs_bg_threshold.setEnabled(False)

            # Log transform
            if self.cb_log.isChecked():
                self.displayedStack = remove_nan_inf(np.log10(self.displayedStack))
                logger.info("Log Stack is in use")

            # Smoothing (removed caching to prevent pipeline issues)
            if self.cb_smooth.isChecked():
                self.hs_smooth_size.setEnabled(True)
                window = self.hs_smooth_size.value()
                if window % 2 == 0:
                    window += 1
                    
                self.displayedStack = smoothen(self.displayedStack, w_size=window)
                    
                self.smooth_winow_size.setText("Window size: " + str(window))
                logger.info("Spectrum Smoothening Applied")
            else:
                self.hs_smooth_size.setEnabled(False)

            # Normalization (fast operation, no caching needed)
            if self.cb_norm.isChecked():
                logger.info("Normalizing spectra")
                self.displayedStack = normalize(self.displayedStack, norm_point=-1)

            logger.info("Updated image is in use")
            
        finally:
            # Always restore cursor
            QApplication.restoreOverrideCursor()

    # ImageView

    def view_stack(self):

        if not self.im_stack.ndim == 3:
            raise ValueError("stack should be an ndarray with ndim == 3")
        else:
            self.update_stack()
            # self.StackUpdateThread()

        try:
            self.image_view.removeItem(self.image_roi_math)
        except Exception:
            pass

        (self.dim1, self.dim2, self.dim3) = self.displayedStack.shape
        self.image_view.setImage(self.displayedStack)
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.setPredefinedGradient("viridis")
        self.image_view.setCurrentIndex(self.dim1 // 2)
        if len(self.energy) == 0:
            self.energy = np.arange(self.z1, self.z2) * 10
            logger.info("Arbitary X-axis used in the plot for XANES")
        self.sz = np.max(
            [int(self.dim2 * 0.1), int(self.dim3 * 0.1)]
        )  # size of the roi set to be 10% of the image area

        self.stack_center = self.energy[len(self.energy) // 2]
        self.stack_width = (self.energy.max() - self.energy.min()) // 10
        self.spec_roi = pg.LinearRegionItem(
            values=(self.stack_center - self.stack_width, self.stack_center + self.stack_width)
        )

        # a second optional ROI for calculations follow
        self.image_roi_math = pg.PolyLineROI(
            [[0, 0], [0, self.sz], [self.sz, self.sz], [self.sz, 0]],
            pos=(int(self.dim3 // 3), int(self.dim2 // 3)),
            pen="r",
            closed=True,
            removable=True,
        )

        self.spec_roi_math = pg.LinearRegionItem(
            values=(self.stack_center - self.stack_width - 10, self.stack_center + self.stack_width - 10),
            pen="r",
            brush=QtGui.QColor(0, 255, 200, 50),
        )
        self.spec_lo_m_idx = self.spec_hi_m_idx = 0

        self.setImageROI()
        self.update_spectrum()
        self.update_image_roi()

        if not self.isAReload:
            # image connections
            self.image_view.mousePressEvent = self.getPointSpectrum
            self.pb_apply_spec_calc.clicked.connect(self.spec_roi_calc)
            self.rb_math_roi.clicked.connect(self.update_spectrum)
            self.pb_add_roi_2.clicked.connect(self.math_img_roi_flag)
            self.image_roi_math.sigRegionChangeFinished.connect(self.image_roi_calc)
            self.pb_apply_img_calc.clicked.connect(self.image_roi_calc)

        self.spec_roi.sigRegionChanged.connect(self.update_image_roi)
        self.spec_roi_math.sigRegionChangeFinished.connect(self.spec_roi_calc)

        [
            rbs.clicked.connect(self.setImageROI)
            for rbs in [self.rb_poly_roi, self.rb_elli_roi, self.rb_rect_roi, self.rb_line_roi, self.rb_circle_roi]
        ]

    def disconnectImageActions(self):
        for btns in [self.pb_apply_spec_calc, self.rb_math_roi, self.pb_add_roi_2, self.pb_apply_img_calc]:
            try:
                btns.disconnect()
            except Exception:
                pass

    def select_elist(self):
        self.energyFileChooser()
        self.efileLoader()
        self.energyUnitCheck()
        self.view_stack()

    def efileLoader(self):

        if self.efilePath:

            if str(self.efilePath).endswith("log_tiff.txt"):
                self.energy = energy_from_logfile(logfile=str(self.efilePath))
                logger.info("Log file from pyxrf processing")

            else:
                self.energy = np.loadtxt(str(self.efilePath))
            self.change_color_on_load(self.pb_elist_xanes)
            logger.info("Energy file loaded")

        else:
            self.statusbar_main.showMessage("No Energy List Selected, Setting Arbitary Axis")
            self.energy = np.arange(self.im_stack.shape[0])
            logger.info("Arbitary Energy Axis")

        # assert len(self.energy) == self.dim1, "Number of Energy Points is not equal to stack length"

    def energyUnitCheck(self):

        if np.max(self.energy) < 100:
            self.cb_kev_flag.setChecked(True)
            self.energy *= 1000

        else:
            self.cb_kev_flag.setChecked(False)

    def select_ref_file(self):
        self.pb_xanes_fit.setEnabled(True)
        self.ref_names = []
        file_name = QFileDialog().getOpenFileName(self, "Open reference file", self.user_wd, "text file (*.csv *.nor)")
        if file_name[0]:
            if file_name[0].endswith(".nor"):
                self.refs, self.ref_names = create_df_from_nor_try2(athenafile=file_name[0])
                self.change_color_on_load(self.pb_ref_xanes)

            elif file_name[0].endswith(".csv"):
                self.refs = pd.read_csv(file_name[0])
                self.ref_names = list(self.refs.keys())
                
                self.change_color_on_load(self.pb_ref_xanes)

            self.user_wd = os.path.dirname(file_name[0])

        else:
            logger.error("No file selected")
            pass

        logger.info(f"{self.refs.shape = }")

        self.plt_xanes_refs()

    def plt_xanes_refs(self):

        try:
            self.ref_plot.close()
        except Exception:
            pass

        self.ref_plot = plot(title="Reference Standards")
        self.ref_plot.setLabel("bottom", "Energy")
        self.ref_plot.setLabel("left", "Intensity")
        self.ref_plot.addLegend()

        for n in range(np.shape(self.refs)[1]):

            if not n == 0:
                self.ref_plot.plot(
                    self.refs.values[:, 0],
                    self.refs.values[:, n],
                    pen=pg.mkPen(self.plt_colors[n - 1], width=self.plotWidth),
                    name=self.ref_names[n],
                )

    def getPointSpectrum(self, event):
        if event.type() == QtCore.QEvent.Type.MouseButtonDblClick:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.xpixel = int(self.image_view.view.mapSceneToView(event.pos().toPointF()).x()) - 1
                zlim, ylim, xlim = self.displayedStack.shape

                if self.xpixel > xlim:
                    self.xpixel = xlim - 1

                self.ypixel = int(self.image_view.view.mapSceneToView(event.pos().toPointF()).y()) - 1
                if self.ypixel > ylim:
                    self.ypixel = ylim - 1
                self.spectrum_view.addLegend()
                self.point_spectrum = self.displayedStack[:, self.ypixel, self.xpixel]
                self.spectrum_view.plot(
                    self.xdata,
                    self.point_spectrum,
                    clear=True,
                    pen=pg.mkPen(pg.mkColor(0, 0, 255, 255), width=self.plotWidth),
                    symbol="o",
                    symbolSize=6,
                    symbolBrush="r",
                    name=f"Point Spectrum; x= {self.xpixel}, y= {self.ypixel}",
                )

                self.spectrum_view.addItem(self.spec_roi)

                self.statusbar_main.showMessage(f"{self.xpixel} and {self.ypixel}")

    def setImageROI(self):

        self.lineROI = pg.LineSegmentROI([[int(self.dim3 // 2), int(self.dim2 // 2)], [self.sz, self.sz]], pen="r")

        self.rectROI = pg.RectROI(
            [int(self.dim3 // 2), int(self.dim2 // 2)],
            [self.sz, self.sz],
            pen="w",
            maxBounds=QtCore.QRectF(0, 0, self.dim3, self.dim2),
        )

        self.rectROI.addTranslateHandle([0, 0], [2, 2])
        self.rectROI.addRotateHandle([0, 1], [2, 2])

        self.ellipseROI = pg.EllipseROI(
            [int(self.dim3 // 2), int(self.dim2 // 2)],
            [self.sz, self.sz],
            pen="w",
            maxBounds=QtCore.QRectF(0, 0, self.dim3, self.dim2),
        )

        self.circleROI = pg.CircleROI(
            [int(self.dim3 // 2), int(self.dim2 // 2)],
            [self.sz, self.sz],
            pen="w",
            maxBounds=QtCore.QRectF(0, 0, self.dim3, self.dim2),
        )  # pos and size

        self.polyLineROI = pg.PolyLineROI(
            [[0, 0], [0, self.sz], [self.sz, self.sz], [self.sz, 0]],
            pos=(int(self.dim3 // 2), int(self.dim2 // 2)),
            maxBounds=QtCore.QRectF(0, 0, self.dim3, self.dim2),
            closed=True,
            removable=True,
        )

        self.rois = {
            "rb_line_roi": self.lineROI,
            "rb_rect_roi": self.rectROI,
            "rb_circle_roi": self.circleROI,
            "rb_elli_roi": self.ellipseROI,
            "rb_poly_roi": self.polyLineROI,
        }

        button_name = self.sender()

        if button_name.objectName() in self.rois.keys():
            self.roi_preference = button_name.objectName()

        else:
            self.roi_preference = "rb_rect_roi"  # default

        try:
            self.image_view.removeItem(self.image_roi)

        except Exception:
            pass

        # ROI settings for image, used polyline roi with non rectangular shape

        self.image_roi = self.rois[self.roi_preference]
        self.image_view.addItem(self.image_roi)
        self.image_roi.sigRegionChanged.connect(self.update_spectrum)

    def replot_image(self):
        self.update_stack()
        self.update_spectrum()
        self.update_image_roi()

    def update_spec_roi_values(self):
        self.stack_center = int(self.energy[len(self.energy) // 2])
        self.stack_width = int((self.energy.max() - self.energy.min()) * 0.05)
        self.spec_roi.setBounds([self.xdata[0], self.xdata[-1]])  # if want to set bounds for the spec roi
        self.spec_roi_math.setBounds([self.xdata[0], self.xdata[-1]])

    def update_spectrum(self):

        # set x-axis values; array taken from energy values, if clipped z box values will update the array
        self.xdata = self.energy[self.sb_zrange1.value() : self.sb_zrange2.value()]

        # get the cropped stack from ROI region; pyqtgraph function is used
        self.roi_img_stk = self.image_roi.getArrayRegion(
            self.displayedStack, self.image_view.imageItem, axes=(1, 2)
        )

        posx, posy = self.image_roi.pos()
        self.le_roi.setText(str(int(posx)) + ":" + str(int(posy)))

        # display the ROI features in the line edit boxes
        if self.roi_img_stk.ndim == 3:
            sizex, sizey = self.roi_img_stk.shape[1], self.roi_img_stk.shape[2]
            self.le_roi_size.setText(str(sizex) + "," + str(sizey))
            self.mean_spectra = get_mean_spectra(self.roi_img_stk)

        elif self.roi_img_stk.ndim == 2:
            sizex, sizey = self.roi_img_stk.shape[0], self.roi_img_stk.shape[1]
            self.le_roi_size.setText(str(sizex) + "," + str(sizey))
            self.mean_spectra = self.roi_img_stk.mean(-1)

        self.spectrum_view.addLegend()

        try:
            self.spectrum_view.plot(
                self.xdata,
                self.mean_spectra,
                pen=pg.mkPen(pg.mkColor(5, 255, 5, 255), width=self.plotWidth),
                clear=True,
                symbol="o",
                symbolSize=6,
                symbolBrush="r",
                name="ROI Spectrum",
            )
        except Exception:
            self.spectrum_view.plot(
                self.mean_spectra,
                clear=True,
                pen=pg.mkPen(pg.mkColor(5, 255, 5, 255), width=self.plotWidth),
                symbol="o",
                symbolSize=6,
                symbolBrush="r",
                name="ROI Spectrum",
            )

        if self.energy[-1] > 1000:
            self.e_unit = "eV"
        else:
            self.e_unit = "keV"

        self.spectrum_view.setLabel("bottom", "Energy", self.e_unit)
        self.spectrum_view.setLabel("left", "Intensity", "A.U.")
        self.spectrum_view.addItem(self.spec_roi)
        self.update_spec_roi_values()
        self.math_roi_flag()

    def update_image_roi(self):
        self.spec_lo, self.spec_hi = self.spec_roi.getRegion()
        self.spec_lo_idx = (np.abs(self.energy - self.spec_lo)).argmin()
        self.spec_hi_idx = (np.abs(self.energy - self.spec_hi)).argmin()
        self.le_spec_roi.setText(str(int(self.spec_lo)) + ":" + str(int(self.spec_hi)))
        self.le_spec_roi_size.setText(str(int(self.spec_hi - self.spec_lo)))
        self.update_spec_roi_values()
        self.stackIndexToNames()

        try:
            if int(self.spec_lo_idx) == int(self.spec_hi_idx):
                self.disp_img = self.displayedStack[int(self.spec_hi_idx), :, :]

            else:
                self.disp_img = self.displayedStack[int(self.spec_lo_idx) : int(self.spec_hi_idx), :, :].mean(0)

            self.image_view.setImage(self.disp_img)
            self.statusbar_main.showMessage(f"Image Display is {self.corrImg1}")
        except Exception:
            logger.warning("Indices are out of range; Image cannot be created")
            pass

    def set_spec_roi(self):
        self.spec_lo_, self.spec_hi_ = int(self.sb_roi_spec_s.value()), int(self.sb_roi_spec_e.value())
        self.spec_lo_idx_ = (np.abs(self.energy - self.spec_lo_)).argmin()
        self.spec_hi_idx_ = (np.abs(self.energy - self.spec_hi_)).argmin()
        self.spec_roi.setRegion((self.xdata[self.spec_lo_idx_], self.xdata[self.spec_hi_idx_]))
        self.update_image_roi()

    def math_roi_flag(self):
        if self.rb_math_roi.isChecked():
            self.rb_math_roi.setStyleSheet("color : green")
            self.spectrum_view.addItem(self.spec_roi_math)
        else:
            self.spectrum_view.removeItem(self.spec_roi_math)

    def spec_roi_calc(self):

        self.spec_lo_m, self.spec_hi_m = self.spec_roi_math.getRegion()
        self.spec_lo_m_idx = (np.abs(self.energy - self.spec_lo_m)).argmin()
        self.spec_hi_m_idx = (np.abs(self.energy - self.spec_hi_m)).argmin()

        if int(self.spec_lo_idx) == int(self.spec_hi_idx):
            self.img1 = self.displayedStack[int(self.spec_hi_idx), :, :]

        else:
            self.img1 = self.displayedStack[int(self.spec_lo_idx) : int(self.spec_hi_idx), :, :].mean(0)

        if int(self.spec_lo_m_idx) == int(self.spec_hi_m_idx):
            self.img2 = self.displayedStack[int(self.spec_hi_m_idx), :, :]

        else:
            self.img2 = self.displayedStack[int(self.spec_lo_m_idx) : int(self.spec_hi_m_idx), :, :].mean(0)

        if self.cb_roi_operation.currentText() == "Correlation Plot":
            self.correlation_plot()

        else:
            calc = {"Divide": np.divide, "Subtract": np.subtract, "Add": np.add}
            self.disp_img = remove_nan_inf(calc[self.cb_roi_operation.currentText()](self.img1, self.img2))
            self.image_view.setImage(self.disp_img)

    def math_img_roi_flag(self):

        button_name = self.sender().text()
        logger.info(f"{button_name}")
        if button_name == "Add ROI_2":
            self.image_view.addItem(self.image_roi_math)
            self.pb_add_roi_2.setText("Remove ROI_2")
            self.image_roi2_flag = 1
        elif button_name == "Remove ROI_2":
            self.image_view.removeItem(self.image_roi_math)
            self.pb_add_roi_2.setText("Add ROI_2")
            self.image_roi2_flag = 0

        else:
            pass
            logger.error("Unknown signal for second ROI")

    def image_roi_calc(self):

        if self.image_roi2_flag == 1:
            self.calc = {"Divide": np.divide, "Subtract": np.subtract, "Add": np.add}
            self.update_spec_image_roi()
        else:
            logger.error("No ROI2 found")
            return

    def update_spec_image_roi(self):

        self.math_roi_reg = self.image_roi_math.getArrayRegion(
            self.displayedStack, self.image_view.imageItem, axes=(1, 2)
        )
        if self.math_roi_reg.ndim == 3:

            self.math_roi_spectra = get_mean_spectra(self.math_roi_reg)

        elif self.roi_img_stk.ndim == 2:
            self.math_roi_spectra = self.math_roi_reg.mean(-1)

        if self.cb_img_roi_action.currentText() in self.calc.keys():

            calc_spec = self.calc[self.cb_img_roi_action.currentText()](self.mean_spectra, self.math_roi_spectra)
            self.spectrum_view.addLegend()
            self.spectrum_view.plot(
                self.xdata,
                calc_spec,
                clear=True,
                pen=pg.mkPen("m", width=2),
                name=self.cb_img_roi_action.currentText() + "ed",
            )
            self.spectrum_view.plot(self.xdata, self.math_roi_spectra, pen=pg.mkPen("y", width=2), name="ROI2")
            self.spectrum_view.plot(self.xdata, self.mean_spectra, pen=pg.mkPen("g", width=2), name="ROI1")

        elif self.cb_img_roi_action.currentText() == "Compare":
            self.spectrum_view.plot(
                self.xdata, self.math_roi_spectra, pen=pg.mkPen("y", width=2), clear=True, name="ROI2"
            )
            self.spectrum_view.plot(self.xdata, self.mean_spectra, pen=pg.mkPen("g", width=2), name="ROI1")

        self.spectrum_view.addItem(self.spec_roi)

    def displayStackInfo(self):

        try:

            if isinstance(self.file_name, list):
                info = f"Folder; {os.path.dirname(self.file_name[0])} \n"
                for n, name in enumerate(self.file_name):
                    info += f"{n}: {os.path.basename(name)} \n"

                # info = f'Stack order; {[name for name in enumerate(self.file_name)]}'
            else:
                info = f"Stack; {self.file_name}"

            self.infoWindow = StackInfo(str(info))
            self.infoWindow.show()

        except AttributeError:
            self.statusbar_main.showMessage("Warning: No Image Data Loaded")

    def stackIndexToNames(self):
        # create list of tiff file names for virtutal stack for plot axes
        self.elemFileName = []

        if isinstance(self.file_name, list):
            for name in self.file_name:
                self.elemFileName.append(os.path.basename(name).split(".")[0])

            logger.info(f" Virtual Stack - list of image names; {self.elemFileName}")

            # if the roi focus on one frame, Note that this slicing excludes the last index
            if int(self.spec_lo_idx) == int(self.spec_hi_idx):
                self.corrImg1 = str(self.elemFileName[int(self.spec_lo_idx)])
            else:
                self.corrImg1 = self.elemFileName[int(self.spec_lo_idx) : int(self.spec_hi_idx)]
                if len(self.corrImg1) > 1:
                    self.corrImg1 = f"Sum of {self.corrImg1} "

            if int(self.spec_lo_m_idx) == int(self.spec_hi_m_idx):
                self.corrImg2 = str(self.elemFileName[int(self.spec_lo_m_idx)])

            else:
                self.corrImg2 = self.elemFileName[int(self.spec_lo_m_idx) : int(self.spec_hi_m_idx)]

                if len(self.corrImg2) > 1:
                    self.corrImg2 = f"Sum of {self.corrImg2}"

            logger.info(
                f"Correlation stack {int(self.spec_lo_idx)}:{int(self.spec_hi_idx)} with "
                f"{int(self.spec_lo_m_idx)}:{int(self.spec_hi_m_idx)}"
            )

            logger.info(f" Virtual Stack; corrlation plot of {self.corrImg1} vs {self.corrImg2}")
        else:
            self.corrImg1 = (
                f" Sum of {os.path.basename(self.file_name).split('.')[0]}_{int(self.spec_lo_idx)} "
                f"to {int(self.spec_hi_idx)}"
            )
            self.corrImg2 = (
                f" Sum of {os.path.basename(self.file_name).split('.')[0]}_{int(self.spec_lo_m_idx)} "
                f"to {int(self.spec_hi_m_idx)}"
            )
            # logger.info(f" corrlation plot of {self.corrImg1} vs {self.corrImg2}")

    def stackToCSV(self):

        self.stackIndexToNames()
        self.imageDf = pd.DataFrame()
        if len(self.elemFileName) == len(self.displayedStack):
            for name, image in zip(self.elemFileName, self.displayedStack):
                self.imageDf[f'{name}'] = image.flatten()
        # print(self.imageDf.head())
        else:
            self.imageDf = image_to_pandas2(self.displayedStack)

        file_name = QFileDialog().getSaveFileName(self, 
                                                  "Save CSV Data", 
                                                  os.path.join(self.user_wd,'image_2DArray.csv'), 
                                                  'file (*csv)')
        if file_name[0]:
            self.imageDf.to_csv(path_or_buf=file_name[0])
            self.user_wd = os.path.dirname(file_name[0])
            self.statusbar_main.showMessage(f"Data saved to {file_name[0]}")
        else:
            pass

    def correlation_plot(self):
        self.stackIndexToNames()

        self.statusbar_main.showMessage(f"Correlation of {self.corrImg1} with {self.corrImg2}")

        if self.rb_roiRegionOnly.isChecked():
            self.roi_mask = self.image_roi.getArrayRegion(
                self.displayedStack, self.image_view.imageItem, axes=(1, 2)
            )
            self.roi_img1 = np.mean(self.roi_mask[int(self.spec_lo_idx) : int(self.spec_hi_idx)], axis=0)
            self.roi_img2 = np.mean(self.roi_mask[int(self.spec_lo_m_idx) : int(self.spec_hi_m_idx)], axis=0)
            self.scatter_window = ScatterPlot(
                self.roi_img1, self.roi_img2, (str(self.corrImg1), str(self.corrImg2))
            )

        else:

            self.scatter_window = ScatterPlot(self.img1, self.img2, (str(self.corrImg1), str(self.corrImg2)))

        self.scatter_window.show()

    def plotCorrelationsAllCombinations(self):

        print("Plotting all correlations")
        self.stackIndexToNames()
        allElemCombNum = list(combinations(np.arange(len(self.elemFileName)), 2))

        self.scW1 = self.scW2 = self.scW3 = self.scW4 = self.scW5 = None
        self.scW6 = self.scW7 = self.scW8 = self.scW9 = self.scW10 = None

        self.scWindowList = [
            self.scW1,
            self.scW2,
            self.scW3,
            self.scW4,
            self.scW5,
            self.scW6,
            self.scW7,
            self.scW8,
            self.scW9,
            self.scW10,
        ]
        self.scWindowDict = {
            1: self.scW1,
            2: self.scW2,
            3: self.scW3,
            4: self.scW4,
            5: self.scW5,
            6: self.scW6,
            7: self.scW7,
            8: self.scW8,
            9: self.scW9,
            10: self.scW10,
        }

        if len(allElemCombNum) > len(self.scWindowDict):

            reply = QMessageBox.warning(
                self,
                "Plot Window Limit",
                f"The number of combination exceeds "
                f"maxiumum number of "
                f"plot windows. First {len(self.scWindowDict)} "
                f"combinations will be plotted. \n      Proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:

                for i, pair in enumerate(allElemCombNum):
                    im1 = self.displayedStack[pair[0]]
                    im2 = self.displayedStack[pair[1]]
                    im1Name = self.elemFileName[pair[0]]
                    im2Name = self.elemFileName[pair[1]]

                    self.scWindowDict[i] = ScatterPlot(im1, im2, (str(im1Name), str(im2Name)))
                    self.scWindowDict[i].show()

            if reply == QMessageBox.StandardButton.No:
                return

        else:

            for i, pair in enumerate(allElemCombNum):
                im1 = self.displayedStack[pair[0]]
                im2 = self.displayedStack[pair[1]]
                im1Name = self.elemFileName[pair[0]]
                im2Name = self.elemFileName[pair[1]]

                self.scWindowDict[i] = ScatterPlot(im1, im2, (str(im1Name), str(im2Name)))
                self.scWindowDict[i].show()

    def getROIMask(self):
        self.roi_mask = self.image_roi.getArrayRegion(self.displayedStack, self.image_view.imageItem, axes=(1, 2))
        self.newWindow = singleStackViewer(self.roi_mask)
        self.newWindow.show()

    def save_stack(self, method="raw"):

        # self.update_stack()
        file_name = QFileDialog().getSaveFileName(
            self, 
            "Save image data", 
            os.path.join(self.user_wd,"image_data.tiff"), 
            "image file(*tiff *tif )"
        )
        if file_name[0]:
            if method == "raw":

                tf.imwrite(file_name[0], self.displayedStack)
                logger.info(f"Updated Image Saved: {file_name[0]}")
                self.statusbar_main.showMessage(f"Updated Image Saved: {file_name[0]}")
            elif method == "sum":
                tf.imwrite(file_name[0], np.sum(self.displayedStack, axis=0))

            elif method == "mean":
                tf.imwrite(file_name[0], np.mean(self.displayedStack, axis=0))

            self.user_wd = os.path.dirname(file_name[0])

        else:
            self.statusbar_main.showMessage("Saving cancelled")
            logger.info(f"Save failed: {file_name[0]}")
            pass

    def save_disp_img(self):
        file_name = QFileDialog().getSaveFileName(self, 
                                                  "Save image data",
                                                    os.path.join(self.user_wd,"image.tiff"), 
                                                    "image file(*tiff *tif )")
        if file_name[0]:
            tf.imwrite(file_name[0], self.disp_img)
            self.statusbar_main.showMessage(f"Image Saved to {file_name[0]}")
            self.user_wd = os.path.dirname(file_name[0])
            logger.info(f"Updated Image Saved: {file_name[0]}")

        else:
            logger.error("No file to save")
            self.statusbar_main.showMessage("Saving cancelled")
            pass

    def getLivePlotData(self):
        try:

            data = np.squeeze([c.getData() for c in self.spectrum_view.plotItem.curves])
            # print(np.shape(data))
            if data.ndim == 2:
                self.mu_ = data[1]
                self.e_ = data[0]
            elif data.ndim == 3:
                e_mu = data[0, :, :]
                self.mu_ = e_mu[1]
                self.e_ = e_mu[0]

            else:
                logger.error(f" Data shape of {data.ndim} is not supported")
                pass
        except AttributeError:
            logger.error("No data loaded")
            pass

    def addSpectrumToCollector(self):
        self.getLivePlotData()
        self.spectrum_view_collect.plot(self.e_, self.mu_, name="ROI Spectrum")
        self.spectrum_view_collect.setLabel("bottom", "Energy", self.e_unit)
        self.spectrum_view_collect.setLabel("left", "Intensity", "A.U.")

    def findEo(self):
        try:
            self.getLivePlotData()
            e0_init = self.e_[np.argmax(np.gradient(self.mu_))]
            self.dsb_norm_Eo.setValue(e0_init)

        except AttributeError:
            logger.error("No data loaded")
            pass

    def initNormVals(self):
        self.getLivePlotData()
        e0_init = self.e_[np.argmax(np.gradient(self.mu_))]
        pre_edge, post_edge, norm = xanesNormalization(
            self.e_,
            self.mu_,
            e0=e0_init,
            nnorm=1,
            nvict=0,
        )
        # xanesNormalization now returns arrays, not scalar bounds
        # Set reasonable default bounds for the UI based on standard XANES practice
        self.dsb_norm_pre1.setValue(-50)  # 50 eV before edge
        self.dsb_norm_pre2.setValue(-20)  # 20 eV before edge
        self.dsb_norm_post1.setValue(50)   # 50 eV after edge
        self.dsb_norm_post2.setValue(150)  # 150 eV after edge
        self.dsb_norm_Eo.setValue(e0_init)

    def getNormParams(self):
        self.getLivePlotData()
        eo_ = self.dsb_norm_Eo.value()
        pre1_, pre2_ = self.dsb_norm_pre1.value(), self.dsb_norm_pre2.value()
        norm1_, norm2_ = self.dsb_norm_post1.value(), self.dsb_norm_post2.value()
        norm_order = self.sb_norm_order.value()

        return eo_, pre1_, pre2_, norm1_, norm2_, norm_order

    def exportNormParams(self):
        self.xanesNormParam = {}
        eo_, pre1_, pre2_, norm1_, norm2_, norm_order = self.getNormParams()
        self.xanesNormParam["E0"] = eo_
        self.xanesNormParam["pre1"] = pre1_
        self.xanesNormParam["pre2"] = pre2_
        self.xanesNormParam["post1"] = norm1_
        self.xanesNormParam["post2"] = norm2_
        self.xanesNormParam["norm_order"] = norm_order

        file_name = QtWidgets.QFileDialog().getSaveFileName(
            self, 
            "Save XANES Norm Params", 
            os.path.join(self.user_wd,"xanes_norm_params.csv"), 
            "csv file(*csv)"
        )
        if file_name[0]:
            pd.DataFrame(self.xanesNormParam, index=[0]).to_csv(file_name[0])
            self.user_wd = os.path.dirname(file_name[0])

        else:
            pass

    def importNormParams(self):

        file_name = QtWidgets.QFileDialog().getOpenFileName(
            self, "Open a XANES Norm File", self.user_wd, "csv file(*csv);;all_files (*)"
        )

        if file_name[0]:
            xanesNormParam = pd.read_csv(file_name[0])
            self.dsb_norm_Eo.setValue(xanesNormParam["E0"])
            self.dsb_norm_pre1.setValue(xanesNormParam["pre1"])
            self.dsb_norm_pre2.setValue(xanesNormParam["pre2"])
            self.dsb_norm_post1.setValue(xanesNormParam["post1"])
            self.dsb_norm_post2.setValue(xanesNormParam["post2"])
            self.sb_norm_order.setValue(xanesNormParam["norm_order"])
            self.user_wd = os.path.dirname(file_name[0])

    def nomalizeLiveSpec(self):
        eo_, pre1_, pre2_, norm1_, norm2_, norm_order = self.getNormParams()
        self.spectrum_view.clear()
        colors = np.array(("c", "r", "m"))

        if self.cb_mback.isChecked():
            pass
            # f2, normXANES = xanesNormalization(
            #     self.e_,
            #     self.mu_,
            #     e0=eo_,
            #     nnorm=norm_order,
            #     nvict=0,
            #     pre1=pre1_,
            #     pre2=pre2_,
            #     norm1=norm1_,
            #     norm2=norm2_,
            #     useFlattened=self.cb_xanes_flat.isChecked(),
            # )

            # names = np.array(("matched mu(E)", "f2"))
            # data_array = np.array((normXANES, f2))

        else:
            pre_line, post_line, normXANES = xanesNormalization(
                self.e_,
                self.mu_,
                e0=eo_,
                nnorm=norm_order,
                nvict=0,
                pre1=pre1_,
                pre2=pre2_,
                norm1=norm1_,
                norm2=norm2_,
                useFlattened=self.cb_xanes_flat.isChecked()
            )

            names = np.array(("Spectrum", "Pre", "Post"))
            data_array = np.array((self.mu_, pre_line, post_line))


        for data, clr, name in zip(data_array, colors, names):
            self.spectrum_view.plot(self.e_, data, pen=pg.mkPen(clr, width=self.plotWidth), name=name)

        self.spectrum_view_norm.plot(
            self.e_, normXANES, clear=True, pen=pg.mkPen(self.plt_colors[-1], width=self.plotWidth))

        self.spectrum_view_norm.setLabel("bottom", "Energy", self.e_unit)
        self.spectrum_view_norm.setLabel("left", "Norm. Intensity", "A.U.")

    def normalizeStack(self):
        self.getLivePlotData()
        eo_, pre1_, pre2_, norm1_, norm2_, norm_order = self.getNormParams()

        self.im_stack = self.displayedStack = xanesNormStack(
            self.e_,
            self.displayedStack,
            e0=eo_,
            nnorm=norm_order,
            nvict=0,
            pre1=pre1_,
            pre2=pre2_,
            norm1=norm1_,
            norm2=norm2_,
            ignorePostEdgeNorm=self.cb_xanes_postedge.isChecked()
        )
        # self.im_stack = self.displayedStack

    def transposeStack(self):
        self.im_stack = self.displayedStack = np.transpose(self.displayedStack, (2, 1, 0))

    def swapStackXY(self):
        self.im_stack = self.displayedStack = np.transpose(self.displayedStack, (0, 2, 1))

    def removeROIBGStack(self):
        self.displayedStack = subtractBackground(self.displayedStack, self.mean_spectra)

    def resetCollectorSpec(self):
        pass

    def saveCollectorPlot(self):
        exporter = pg.exporters.CSVExporter(self.spectrum_view_collect.plotItem)
        #exporter.parameters()["columnMode"] = "(x,y,y,y) for all plots"
        file_name = QFileDialog().getSaveFileName(self, "save spectra", self.user_wd, "spectra (*csv)")
        if file_name[0]:
            exporter.export(file_name[0] + ".csv")
            self.user_wd = os.path.dirname(file_name[0])
        else:
            self.statusbar_main.showMessage("Saving cancelled")
            pass

    def save_disp_spec(self):

        exporter = pg.exporters.CSVExporter(self.spectrum_view.plotItem)
        #exporter.parameters()["columnMode"] = "(x,y,y,y) for all plots"
        file_name = QFileDialog().getSaveFileName(self, 
                                                  "save spectrum", 
                                                  os.path.join(self.user_wd,"spectrum.csv"), 
                                                  "spectra (*csv)")
        if file_name[0]:
            exporter.export(file_name[0] + ".csv")
            self.user_wd = os.path.dirname(file_name[0])
        else:
            self.statusbar_main.showMessage("Saving cancelled")
            pass

    def saveEnergyList(self):
        file_name = QFileDialog().getSaveFileName(self, 
                                                  "save energy list", 
                                                  os.path.join(self.user_wd,"energy_list.txt"), 
                                                  "text file (*txt)")
        if file_name[0]:
            np.savetxt(file_name[0], self.xdata, fmt="%.4f")
            self.user_wd = os.path.dirname(file_name[0])
        else:
            pass

    def pca_scree_(self):
        logger.info("Process started..")
        self.update_stack()
        var = pca_scree(self.displayedStack)

        pca_scree_plot = pg.plot(
            var[:24], title="Scree Plot", pen=pg.mkPen("y", width=2, style=QtCore.Qt.DotLine), symbol="o"
        )
        pca_scree_plot.addLine(y=0)
        pca_scree_plot.setLabel("bottom", "Component Number")
        pca_scree_plot.setLabel("left", "Singular Values")

        logger.info("Process complete")

    def calc_comp_(self):

        logger.info("Process started..")

        # self.update_stack()
        n_components = self.sb_ncomp.value()
        method_ = self.cb_comp_method.currentText()

        ims, comp_spec, decon_spec, decomp_map = decompose_stack(
            self.displayedStack, decompose_method=method_, n_components_=n_components
        )

        self._new_window3 = ComponentViewer(ims, self.xdata, comp_spec, decon_spec, decomp_map)
        self._new_window3.show()

        logger.info("Process complete")

    def kmeans_elbow(self):
        logger.info("Process started..")
        # self.update_stack()

        with pg.BusyCursor():
            try:
                clust_n, var = kmeans_variance(self.displayedStack)
                kmeans_var_plot = pg.plot(
                    clust_n, var, title="KMeans Variance", pen=pg.mkPen("y", width=2, style=QtCore.Qt.DotLine),
                    symbol="o"
                )
                kmeans_var_plot.setLabel("bottom", "Cluster Number")
                kmeans_var_plot.setLabel("left", "Sum of squared distances")
                logger.info("Process complete")
            except OverflowError:
                pass
                logger.error("Overflow Error, values are too long")

    def kmeans_elbow_Thread(self):
        # Pass the function to execute
        worker = Worker(self.kmeans_elbow)  # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        # Execute
        self.threadpool.start(worker)

    def clustering_(self):

        logger.info("Process started..")
        # self.update_stack()
        method_ = self.cb_clust_method.currentText()

        decon_images, X_cluster, decon_spectra = cluster_stack(
            self.displayedStack,
            method=method_,
            n_clusters_=self.sb_ncluster.value(),
            decomposed=False,
            decompose_method=self.cb_comp_method.currentText(),
            decompose_comp=self.sb_ncomp.value(),
        )

        self._new_window4 = ClusterViewer(decon_images, self.xdata, X_cluster, decon_spectra)
        self._new_window4.show()

        logger.info("Process complete")

    def change_color_on_load(self, button_name):
        button_name.setStyleSheet("background-color : rgb(0,150,0);" "color: rgb(255,255,255)")

    def energyFileChooser(self):
        file_name = QFileDialog().getOpenFileName(self, 
                                                  "Open energy list", 
                                                  self.user_wd, 
                                                  "text file (*.txt)")
        self.efilePath = file_name[0]

    def fast_xanes_fitting(self):

        self._new_window5 = XANESViewer(self.displayedStack, self.xdata, self.refs, self.ref_names)
        self._new_window5.show()

    # Thread Signals

    def print_output(self, s):
        print(s)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Window Close",
            "Are you sure you want to close?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
            QApplication.closeAllWindows()
        else:
            event.ignore()


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    Supported signals are:
    - finished: No data
    - error:`tuple` (exctype, value, traceback.format_exc() )
    - result: `object` data returned from processing, anything
    - progress: `tuple` indicating progress metadata
    """

    start = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    """
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    """

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """
        # Retrieve args/kwargs here; and fire processing using them
        self.signals.start.emit()
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done




class ComponentViewer(QtWidgets.QMainWindow):
    def __init__(self, comp_stack, energy, comp_spectra, decon_spectra, decomp_map):
        super(ComponentViewer, self).__init__()

        # Load the UI Page
        uic.loadUi(os.path.join(ui_dir, "ComponentView.ui"), self)
        self.centralwidget.setStyleSheet(open(os.path.join(ui_dir, "css/defaultStyle.css")).read())
        self.user_wd = os.path.abspath("~")
       
        self.comp_stack = comp_stack
        self.energy = energy
        self.comp_spectra = comp_spectra
        self.decon_spectra = decon_spectra
        self.decomp_map = decomp_map

        (self.dim1, self.dim3, self.dim2) = self.comp_stack.shape
        self.hs_comp_number.setMaximum(self.dim1 - 1)

        self.image_view.setImage(self.comp_stack)
        self.image_view.setPredefinedGradient("viridis")
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()

        self.image_view2.setImage(self.decomp_map)
        self.image_view2.setPredefinedGradient("bipolar")
        self.image_view2.ui.menuBtn.hide()
        self.image_view2.ui.roiBtn.hide()

        # connection
        self.update_image()
        self.pb_show_all.clicked.connect(lambda:self.show_all_spec(norm_to_max = True, add_offset = True))
        self.hs_comp_number.valueChanged.connect(self.update_image)
        self.actionSave.triggered.connect(self.save_comp_data)
        self.pb_openScatterPlot.clicked.connect(self.openScatterPlot)
        self.pb_showMultiColor.clicked.connect(lambda: self.generateMultiColorView(withSpectra=False))
        self.pb_showMultiImageXANESView.clicked.connect(lambda: self.generateMultiColorView(withSpectra=True))

    def update_image(self):
        im_index = self.hs_comp_number.value()
        self.spectrum_view.setLabel("bottom", "Energy")
        self.spectrum_view.setLabel("left", "Intensity", "A.U.")
        self.spectrum_view.plot(self.energy, self.decon_spectra[:, im_index], clear=True)
        self.component_view.setLabel("bottom", "Energy")
        self.component_view.setLabel("left", "Weight", "A.U.")
        self.component_view.plot(self.energy, self.comp_spectra[:, im_index], clear=True)
        self.label_comp_number.setText(f"{im_index + 1}/{self.dim1}")
        # self.image_view.setCurrentIndex(im_index-1)
        self.image_view.setImage(self.comp_stack[im_index])

    def openScatterPlot(self):
        self.scatter_window = ComponentScatterPlot(self.comp_stack, self.comp_spectra)

        # ph = self.geometry().height()
        # pw = self.geometry().width()
        # px = self.geometry().x()
        # py = self.geometry().y()
        # dw = self.scatter_window.width()
        # dh = self.scatter_window.height()
        # self.scatter_window.setGeometry(px+0.65*pw, py + ph - 2*dh-5, dw, dh)
        self.scatter_window.show()

    def show_all_spec(self, norm_to_max = True, add_offset = True):
        self.spectrum_view.clear()
        self.plt_colors = ["g", "b", "r", "c", "m", "y", "w"] * 10
        offsets = np.arange(0, 2, 0.2)
        self.spectrum_view.addLegend()
        for ii in range(self.decon_spectra.shape[1]):
            to_plot = self.decon_spectra[:, ii]
            if norm_to_max:
                to_plot = self.decon_spectra[:, ii] / self.decon_spectra[:, ii].max()
            if add_offset:
                to_plot = to_plot+ + offsets[ii]

            self.spectrum_view.plot(
                self.energy,
                to_plot,
                pen=self.plt_colors[ii],
                name="component" + str(ii + 1),
            )
        self.component_view.clear()
        self.component_view.addLegend()
        for ii in range(self.comp_spectra.shape[1]):
            to_plot = self.comp_spectra[:, ii]
            if norm_to_max:
                to_plot = self.comp_spectra[:, ii] / self.comp_spectra[:, ii].max()
            if add_offset:
                to_plot = to_plot+ + offsets[ii]
            self.component_view.plot(
                self.energy,
                to_plot,
                pen=self.plt_colors[ii],
                name="eigen_vector" + str(ii + 1),
            )

    def save_comp_data(self):
        file_name = QFileDialog().getSaveFileName(self, "save all data", self.user_wd, "data(*tiff *tif *txt *png )")
        if file_name[0]:
            self.show_all_spec(norm_to_max = False, add_offset = False)
            tf.imwrite(file_name[0] + "_components.tiff", np.float32(self.comp_stack))
            tf.imwrite(file_name[0] + "_component_masks.tiff", np.float32(self.decomp_map))
            exporter_spec = pg.exporters.CSVExporter(self.spectrum_view.plotItem)
            exporter_spec.parameters()["columnMode"] = "(x,y) per plot"
            exporter_spec.export(file_name[0] + "_deconv_spec.csv")
            exporter_eigen = pg.exporters.CSVExporter(self.component_view.plotItem)
            exporter_eigen.parameters()["columnMode"] = "(x,y) per plot"
            exporter_eigen.export(file_name[0] + "_eigen_vectors.csv")
            self.user_wd = os.path.dirname(file_name[0])
        else:
            pass

    def generateMultiColorView(self, withSpectra=False):
        self.multichanneldict = {}

        for n, (colorName, image) in enumerate(zip(cmap_dict.keys(), self.comp_stack.transpose(0, 1, 2))):
            low, high = np.min(image), np.max(image)
            self.multichanneldict[f'Image {n + 1}'] = {'ImageName': f'Image {n + 1}',
                                                       'ImageDir': '.',
                                                       'Image': image,
                                                       'Color': colorName,
                                                       'CmapLimits': (low, high),
                                                       'Opacity': 1.0
                                                       }

        if withSpectra:
            compXanesSpetraAll = pd.DataFrame()
            compXanesSpetraAll['Energy'] = self.energy

            for n, spec in enumerate(self.decon_spectra.T):
                compXanesSpetraAll[f'Component_{n + 1}'] = spec

            self.muli_color_window = MultiXANESWindow(image_dict=self.multichanneldict,
                                                      spec_df=compXanesSpetraAll)
        else:
            self.muli_color_window = MultiChannelWindow(image_dict=self.multichanneldict)

        self.muli_color_window.show()

    # add energy column


class ClusterViewer(QtWidgets.QMainWindow):
    def __init__(self, decon_images, energy, X_cluster, decon_spectra):
        super(ClusterViewer, self).__init__()

        # Load the UI Page
        uic.loadUi(os.path.join(ui_dir, "ClusterView.ui"), self)
        self.user_wd = os.path.abspath("~")
        self.centralwidget.setStyleSheet(open(os.path.join(ui_dir, "css/defaultStyle.css")).read())

        self.decon_images = decon_images
        self.energy = energy
        self.X_cluster = X_cluster
        self.decon_spectra = decon_spectra
        (self.dim1, self.dim3, self.dim2) = self.decon_images.shape
        self.hsb_cluster_number.setMaximum(self.dim1 - 1)
        self.X_cluster = X_cluster

        self.image_view.setImage(self.decon_images, autoHistogramRange=True, autoLevels=True)
        self.image_view.setPredefinedGradient("viridis")
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()

        self.cluster_view.setImage(self.X_cluster, autoHistogramRange=True, autoLevels=True)
        self.cluster_view.setPredefinedGradient("bipolar")
        self.cluster_view.ui.histogram.hide()
        self.cluster_view.ui.menuBtn.hide()
        self.cluster_view.ui.roiBtn.hide()

        # connection
        self.update_display()
        self.hsb_cluster_number.valueChanged.connect(self.update_display)
        self.actionSave.triggered.connect(self.save_clust_data)
        self.pb_show_all_spec.clicked.connect(self.showAllSpec)
        self.pb_showMultiColor.clicked.connect(self.generateMultiColorView)

    def update_display(self):
        im_index = self.hsb_cluster_number.value()
        self.component_view.setLabel("bottom", "Energy")
        self.component_view.setLabel("left", "Intensity", "A.U.")
        self.component_view.plot(self.energy, self.decon_spectra[:, im_index], clear=True)
        # self.image_view.setCurrentIndex(im_index-1)
        self.image_view.setImage(self.decon_images[im_index])
        self.label_comp_number.setText(f"{im_index + 1}/{self.dim1}")

    def save_clust_data(self):
        file_name = QFileDialog().getSaveFileName(self, "", "", "data(*tiff *tif *txt *png )")
        if file_name[0]:

            tf.imwrite(
                file_name[0] + "_cluster.tiff", np.float32(self.decon_images.transpose(0, 2, 1)), imagej=True
            )
            tf.imwrite(file_name[0] + "_cluster_map.tiff", np.float32(self.X_cluster.T), imagej=True)
            np.savetxt(file_name[0] + "_deconv_spec.txt", self.decon_spectra)

        else:
            logger.error("Saving Cancelled")
            self.statusbar.showMessage("Saving Cancelled")
            pass

    def showAllSpec(self):
        self.component_view.clear()
        self.plt_colors = ["g", "b", "r", "c", "m", "y", "w"] * 10
        offsets = np.arange(0, 2, 0.2)
        self.component_view.addLegend()
        for ii in range(self.decon_spectra.shape[1]):
            self.component_view.plot(
                self.energy,
                (self.decon_spectra[:, ii] / self.decon_spectra[:, ii].max()) + offsets[ii],
                pen=self.plt_colors[ii],
                name="cluster" + str(ii + 1),
            )

    def generateMultiColorView(self):
        self.multichanneldict = {}

        for n, (colorName, image) in enumerate(zip(cmap_dict.keys(), self.decon_images.transpose(0, 1, 2))):
            low, high = np.min(image), np.max(image)
            self.multichanneldict[f"Image {n + 1}"] = {
                "ImageName": f"Image {n + 1}",
                "ImageDir": ".",
                "Image": image,
                "Color": colorName,
                "CmapLimits": (low, high),
                "Opacity": 1.0,
            }
        self.muli_color_window = MultiChannelWindow(image_dict=self.multichanneldict)
        self.muli_color_window.show()



class ScatterPlot(QtWidgets.QMainWindow):
    def __init__(self, img1, img2, nameTuple):
        super(ScatterPlot, self).__init__()

        uic.loadUi(os.path.join(ui_dir, "ScatterView.ui"), self)
        self.user_wd = os.path.abspath("~")
        self.centralwidget.setStyleSheet(open(os.path.join(ui_dir, "css/defaultStyle.css")).read())
        self.clearPgPlot()
        self.w1 = self.scatterViewer.addPlot()
        self.img1 = img1
        self.img2 = img2
        self.nameTuple = nameTuple
        x, y = np.shape(self.img1)
        self.s1 = pg.ScatterPlotItem(size=2, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 255))
        # print(self.s1)

        # create three polyline ROIs for masking
        Xsize = self.img1.max() / 6
        Ysize = self.img2.max() / 6

        self.scatter_mask = pg.PolyLineROI(
            [[0, 0], [0, Ysize], [Xsize / 2, Ysize * 1.5], [Xsize, Ysize], [Xsize, 0]],
            pos=None,
            pen=pg.mkPen("r", width=2),
            hoverPen=pg.mkPen("w", width=2),
            closed=True,
            removable=True,
        )

        self.scatter_mask2 = pg.PolyLineROI(
            [
                [Xsize * 1.2, 0],
                [Xsize * 1.2, Ysize * 2],
                [Xsize * 2, Ysize * 2],
                [Xsize * 3, Ysize],
                [Xsize * 2, 0],
            ],
            pos=None,
            pen=pg.mkPen("g", width=2),
            hoverPen=pg.mkPen("w", width=2),
            closed=True,
            removable=True,
        )
        self.scatter_mask3 = pg.PolyLineROI(
            [
                [Xsize * 2.5, 0],
                [Xsize * 2.5, Ysize],
                [Xsize * 4, Ysize],
                [Xsize * 4, 0],
                [Xsize * 3.7, Ysize * -0.5],
            ],
            pos=None,
            pen=pg.mkPen("c", width=2),
            hoverPen=pg.mkPen("w", width=2),
            closed=True,
            removable=True,
        )

        self.fitScatter = self.fitScatter2 = self.fitScatter3 = None

        self.rois = {
            "ROI 1": (self.scatter_mask, self.rb_roi1.isChecked(), self.fitScatter),
            "ROI 2": (self.scatter_mask2, self.rb_roi2.isChecked(), self.fitScatter2),
            "ROI 3": (self.scatter_mask3, self.rb_roi3.isChecked(), self.fitScatter3),
        }

        self.windowNames = {"ROI 1": self.fitScatter, "ROI 2": self.fitScatter2, "ROI 3": self.fitScatter3}

        self.s1.setData(self.img1.flatten(), self.img2.flatten())
        self.w1.setLabel("bottom", self.nameTuple[0], "counts")
        self.label_img1.setText(self.nameTuple[0])
        self.w1.setLabel("left", self.nameTuple[1], "counts")
        self.label_img2.setText(self.nameTuple[1])
        self.w1.addItem(self.s1)

        self.image_view.setImage(self.img1)
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.setPredefinedGradient("thermal")

        self.image_view2.setImage(self.img2)
        self.image_view2.ui.menuBtn.hide()
        self.image_view2.ui.roiBtn.hide()
        self.image_view2.setPredefinedGradient("thermal")

        # connections
        self.actionSave_Plot.triggered.connect(self.pg_export_correlation)
        self.actionSave_Images.triggered.connect(self.tiff_export_images)
        # self.pb_define_mask.clicked.connect(lambda:self.createMask(self.scatter_mask))
        self.pb_define_mask.clicked.connect(self.addMultipleROIs)
        # self.pb_apply_mask.clicked.connect(lambda:self.getMaskRegion(self.scatter_mask))
        self.pb_apply_mask.clicked.connect(self.applyMultipleROIs)
        self.pb_clear_mask.clicked.connect(self.clearMultipleROIs)
        self.pb_compositeScatter.clicked.connect(self.createCompositeScatter)
        [rbs.clicked.connect(self.updateROIDict) for rbs in [self.rb_roi1, self.rb_roi2, self.rb_roi3]]

    def pg_export_correlation(self):

        exporter = pg.exporters.CSVExporter(self.w1)
        #exporter.parameters()["columnMode"] = "(x,y,y,y) for all plots"
        file_name = QFileDialog().getSaveFileName(self, 
                                                  "save correlation", 
                                                  os.path.join(self.user_wd,"correlation.csv"), 
                                                  "spectrum and fit (*csv)")
        if file_name[0]:
            exporter.export(file_name[0] + ".csv")
            self.statusbar.showMessage(f"Data saved to {file_name[0]}")
            self.user_wd = os.path.dirname(file_name[0])
        else:
            pass

    def tiff_export_images(self):
        file_name = QFileDialog().getSaveFileName(self, 
                                                  "save images", 
                                                  os.path.join(self.user_wd,"image.txt"), 
                                                  "spectrum and fit (*tiff)")
        if file_name[0]:
            tf.imwrite(file_name[0] + ".tiff", np.dstack([self.img1, self.img2]).T)
            self.statusbar.showMessage(f"Images saved to {file_name[0]}")
            self.user_wd = os.path.dirname(file_name[0])
        else:
            pass

    def createMask(self, ROIName):

        try:
            self.w1.removeItem(ROIName)
        except Exception:
            pass
        self.w1.addItem(ROIName)

    def clearMask(self, ROIName):
        self.w1.removeItem(ROIName)

    def clearPgPlot(self):
        try:
            self.masked_img.close()
        except Exception:
            pass

    def getMaskRegion(self, ROIName, generateSeperateWindows=True):

        """filter scatterplot points using polylineROI region"""

        # Ref : https://stackoverflow.com/questions/57719303/how-to-map-mouse-position-on-a-scatterplot

        # get the roi region:QPaintPathObject
        roiShape = self.rois[ROIName][0].mapToItem(self.s1, self.rois[ROIName][0].shape())

        # get data in the scatter plot
        scatterData = np.array(self.s1.getData())

        # generate a binary mask for points inside or outside the roishape
        selected = [roiShape.contains(QtCore.QPointF(pt[0], pt[1])) for pt in scatterData.T]

        # reshape the mask to image dimensions
        self.mask2D = np.reshape(selected, (self.img1.shape))

        # get masked image1
        self.maskedImage = self.mask2D * self.img1

        # get rid of the (0,0) values in the masked array
        self.xData, self.yData = np.compress(selected, scatterData[0]), np.compress(selected, scatterData[1])

        # linear regeression of the filtered X,Y data
        result = linregress(self.xData, self.yData)

        # Pearson's correlation of the filtered X,Y data
        pr, pp = stats.pearsonr(self.xData, self.yData)

        # apply the solved equation to xData to generate the fit line
        self.yyData = result.intercept + result.slope * self.xData

        # Prepare strings for fit results and stats
        self.fitLineEqn = (
            f" y =  x*{result.slope :.3e} + {result.intercept :.3e},"
            f"\n R^2 = {result.rvalue**2 :.3f}, r = {pr :.3f}"
        )
        FitStats1 = f" Slope Error = {result.stderr :.3e}, Intercept Error = {result.intercept_stderr :.3e}\n"
        FitStats2 = f" Pearson's correlation coefficient = {pr :.3f}"
        refs = "\n\n ***References****\n\n scipy.stats.linregress, scipy.stats.pearsonr "
        fitStats = (
            f"\n ***{ROIName} Fit Results***\n\n" + " Equation: " + self.fitLineEqn + FitStats1 + FitStats2 + refs
        )

        # generate new window to plot the results

        if generateSeperateWindows:
            self.windowNames[ROIName] = MaskedScatterPlotFit(
                [self.xData, self.yData],
                [self.xData, self.yyData],
                self.mask2D,
                self.maskedImage,
                fitStats,
                self.fitLineEqn,
                self.nameTuple,
            )
            self.windowNames[ROIName].show()

        """
        from scipy.linalg import lstsq
        M = xData[:, np.newaxis]**[0, 1] #use >1 for polynomial fits
        p, res, rnk, s = lstsq(M, yData)
        yyData = p[0] + p[1]*xData
        """

    def updateROIDict(self):
        self.rois = {
            "ROI 1": (self.scatter_mask, self.rb_roi1.isChecked()),
            "ROI 2": (self.scatter_mask2, self.rb_roi2.isChecked()),
            "ROI 3": (self.scatter_mask3, self.rb_roi3.isChecked()),
        }

    def applyMultipleROIs(self):
        with pg.BusyCursor():
            self.updateROIDict()
            for key in self.rois.keys():
                if self.rois[key][1]:
                    self.getMaskRegion(key)
                else:
                    pass

    def addMultipleROIs(self):
        self.updateROIDict()
        for key in self.rois.keys():
            if self.rois[key][1]:
                self.createMask(self.rois[key][0])
            else:
                self.clearMask(self.rois[key][0])

    def clearMultipleROIs(self):
        self.updateROIDict()
        for key in self.rois.keys():
            if not self.rois[key][1]:
                self.clearMask(self.rois[key][0])
            else:
                pass

    def createCompositeScatter(self):

        points = []
        fitLine = []
        masks = []
        roiFitEqn = {}

        self.updateROIDict()
        for n, key in enumerate(self.rois.keys()):
            if self.rois[key][1]:
                self.getMaskRegion(key, generateSeperateWindows=False)
                points.append(np.column_stack([self.xData, self.yData]))
                fitLine.append(np.column_stack([self.xData, self.yyData]))
                masks.append(self.mask2D)
                roiFitEqn[key] = self.fitLineEqn
            else:
                pass

        self.compositeScatterWindow = CompositeScatterPlot(
                                                            points,
                                                            fitLine, 
                                                            np.array(masks), 
                                                            roiFitEqn, 
                                                            self.nameTuple
                                                        )
        self.compositeScatterWindow.show()

    def _createCompositeScatter(self):
        self.scatterColors = ["w", "c", "y", "k", "m"]
        points = []
        fitLine = []

        self.updateROIDict()
        for n, key in enumerate(self.rois.keys()):
            if self.rois[key][1]:
                self.getMaskRegion(key, generateSeperateWindows=False)

                for x, y, yy in zip(self.xData, self.yData, self.yyData):

                    points.append(
                        {
                            "pos": (x, y),
                            "data": "id",
                            "size": 3,
                            "pen": pg.mkPen(None),
                            "brush": self.scatterColors[n],
                        }
                    )
                fitLine.extend(np.column_stack((self.xData, self.yyData)))
            else:
                pass

        logger.info(f" fitline shape: {np.shape(fitLine)}")
        self.compositeScatterWindow = CompositeScatterPlot(points, np.array(fitLine))
        self.compositeScatterWindow.show()

    def getROIParams(self):
        print(np.array(self.scatter_mask.getSceneHandlePositions()))

class MaskedScatterPlotFit(QtWidgets.QMainWindow):
    def __init__(self, scatterData, fitData, mask, maskedImage, fitString, fitEquation, nameTuple):
        super(MaskedScatterPlotFit, self).__init__()

        uic.loadUi(os.path.join(ui_dir, "maskedScatterPlotFit.ui"), self)
        self.user_wd = os.path.abspath("~")
        self.centralwidget.setStyleSheet(open(os.path.join(ui_dir, "css/defaultStyle.css")).read())
        self.scatterData = scatterData
        self.fitData = fitData
        self.mask = mask
        self.maskedImage = maskedImage
        self.fitString = fitString
        self.fitEquation = fitEquation
        self.nameTuple = nameTuple

        # set the graphicslayoutwidget in the ui as canvas
        self.canvas = self.scatterViewer.addPlot()
        self.canvas.addLegend()
        self.canvas.setLabel("bottom", self.nameTuple[0], "counts")
        self.canvas.setLabel("left", self.nameTuple[1], "counts")
        self.gb_maskedImage1.setTitle(f" Masked {self.nameTuple[0]}")

        # generate a scatter plot item
        self.scattered = pg.ScatterPlotItem(size=3.5, pen=pg.mkPen(None), brush=pg.mkBrush(5, 214, 255, 200))

        # set scatter plot data
        self.scattered.setData(scatterData[0], scatterData[1], name="Data")

        # set z value negative to show scatter data behind the fit line
        self.scattered.setZValue(-10)

        # add scatter plot to the canvas
        self.canvas.addItem(self.scattered)

        # generate plotitem for fit line
        self.fitLinePlot = pg.PlotDataItem(pen=pg.mkPen(pg.mkColor(220, 20, 60), width=3.3))

        # set line plot data
        self.fitLinePlot.setData(fitData[0], fitData[1], name="Linear Fit")

        # add line plot to the canvas
        self.canvas.addItem(self.fitLinePlot)

        # display Mask
        self.imageView_mask.setImage(self.mask)
        self.imageView_mask.ui.menuBtn.hide()
        self.imageView_mask.ui.roiBtn.hide()
        self.imageView_mask.setPredefinedGradient("plasma")

        # display masked Image
        self.imageView_maskedImage.setImage(self.maskedImage)
        self.imageView_maskedImage.ui.menuBtn.hide()
        self.imageView_maskedImage.ui.roiBtn.hide()
        self.imageView_maskedImage.setPredefinedGradient("viridis")

        # display Fit stats
        self.text_fit_results.setPlainText(fitString)
        self.canvas.setTitle(self.fitEquation, color="r")

        # connections
        self.pb_copy_results.clicked.connect(self.copyFitResults)
        self.pb_save_results.clicked.connect(self.saveFitResults)
        self.actionSave_Plot.triggered.connect(self.pg_export_correlation)
        self.actionSaveMask.triggered.connect(self.saveMask)
        self.actionSaveMaskedImage.triggered.connect(self.saveImage)

    def saveFitResults(self):
        S__File = QFileDialog.getSaveFileName(self, "save txt", "correlationPlotFit.txt", "txt data (*txt)")

        Text = self.text_fit_results.toPlainText()
        if S__File[0]:
            with open(S__File[0], "w") as file:
                file.write(Text)

    def copyFitResults(self):
        self.text_fit_results.selectAll()
        self.text_fit_results.copy()
        self.statusbar.showMessage("text copied to clipboard")

    def pg_export_correlation(self):

        exporter = pg.exporters.CSVExporter(self.canvas)
        #exporter.parameters()["columnMode"] = "(x,y,y,y) for all plots"
        file_name = QFileDialog().getSaveFileName(
            self, "save correlation", "scatterData.csv", "spectrum and fit (*csv)"
        )
        if file_name[0]:
            exporter.export(file_name[0])
            self.statusbar.showMessage(f"Data saved to {file_name[0]}")
        else:
            pass

    def saveImage(self):

        file_name = QFileDialog().getSaveFileName(self, "Save image data", "image.tiff", "image file(*tiff *tif )")
        if file_name[0]:
            tf.imwrite(file_name[0], self.maskedImage)
            self.statusbar.showMessage(f"Data saved to {file_name[0]}")
        else:
            self.statusbar.showMessage("Saving cancelled")
            pass

    def saveMask(self):

        file_name = QFileDialog().getSaveFileName(self, "Save image data", "mask.tiff", "image file(*tiff *tif )")
        if file_name[0]:
            tf.imwrite(file_name[0], self.mask)
            self.statusbar.showMessage(f"Data saved to {file_name[0]}")
        else:
            self.statusbar.showMessage("Saving cancelled")
            pass

class ComponentScatterPlot(QtWidgets.QMainWindow):
    def __init__(self, decomp_stack, specs):
        super(ComponentScatterPlot, self).__init__()

        uic.loadUi(os.path.join(ui_dir, "ComponentScatterPlot.ui"), self)
        self.user_wd = os.path.abspath("~")
        self.centralwidget.setStyleSheet(open(os.path.join(ui_dir, "css/defaultStyle.css")).read())
        self.w1 = self.scatterViewer.addPlot()
        self.decomp_stack = decomp_stack
        self.specs = specs
        (self.dim1, self.dim3, self.dim2) = self.decomp_stack.shape
        # fill the combonbox depending in the number of components for scatter plot
        for n, combs in enumerate(combinations(np.arange(self.dim1), 2)):
            self.cb_scatter_comp.addItem(str(combs))
            self.cb_scatter_comp.setItemData(n, combs)

        self.s1 = pg.ScatterPlotItem(size=3, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 120))

        self.setImageAndScatterPlot()
        # connections
        self.actionSave_Plot.triggered.connect(self.pg_export_correlation)
        self.actionSave_Images.triggered.connect(self.tiff_export_images)
        self.pb_updateComponents.clicked.connect(self.setImageAndScatterPlot)
        self.pb_define_mask.clicked.connect(self.createMask)
        self.pb_apply_mask.clicked.connect(self.getMaskRegion)
        self.pb_reset_mask.clicked.connect(self.resetMask)
        self.pb_addALine.clicked.connect(lambda: self.createMask(Line=True))

    def setImageAndScatterPlot(self):

        try:
            self.s1.clear()
        except Exception:
            pass

        comp_tuple = self.cb_scatter_comp.currentData()
        self.img1, self.img2 = self.decomp_stack[comp_tuple[0]], self.decomp_stack[comp_tuple[-1]]
        self.image_view.setImage(self.decomp_stack[comp_tuple[0]])
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.setPredefinedGradient("bipolar")

        self.image_view2.setImage(self.decomp_stack[comp_tuple[-1]])
        self.image_view2.ui.menuBtn.hide()
        self.image_view2.ui.roiBtn.hide()
        self.image_view2.setPredefinedGradient("bipolar")

        points = []
        for i, j in zip(self.img1.flatten(), self.img2.flatten()):

            points.append(
                {
                    "pos": (i, j),
                    "data": "id",
                    "size": 5,
                    "pen": pg.mkPen(None),
                    "brush": pg.mkBrush(255, 255, 0, 160),
                }
            )

        self.s1.addPoints(points)
        self.w1.addItem(self.s1)
        # self.s1.setData(self.specs[:, comp_tuple[0]], self.specs[:, comp_tuple[-1]])
        self.w1.setLabel("bottom", f"PC{comp_tuple[0]+1}")
        self.w1.setLabel("left", f"PC{comp_tuple[-1]+1}")
        self.label_im1.setText(f"PC{comp_tuple[0]+1}")
        self.label_im2.setText(f"PC{comp_tuple[-1]+1}")

    def createMask(self, Line=False):

        self.size = self.img1.max() / 10
        self.pos = int(self.img1.mean())

        if Line:
            self.lineROI = pg.LineSegmentROI(
                [0, 1],
                pos=(self.pos, self.pos),
                pen=pg.mkPen("r", width=4),
                hoverPen=pg.mkPen("g", width=4),
                removable=True,
            )
            self.w1.addItem(self.lineROI)

        else:

            self.scatter_mask = pg.PolyLineROI(
                [[0, 0], [0, self.size], [self.size, self.size], [self.size, 0]],
                pos=(self.pos, self.pos),
                pen=pg.mkPen("r", width=4),
                hoverPen=pg.mkPen("g", width=4),
                closed=True,
                removable=True,
            )

            self.w1.addItem(self.scatter_mask)

    def resetMask(self):
        self.clearMask()
        self.createMask()

    def clearMask(self):
        try:
            self.w1.removeItem(self.scatter_mask)
        except AttributeError:
            pass

    def clearPgPlot(self):
        try:
            self.masked_img.close()
        except Exception:
            pass

    def getMaskRegion(self):

        # Ref : https://stackoverflow.com/questions/57719303/how-to-map-mouse-position-on-a-scatterplot

        roiShape = self.scatter_mask.mapToItem(self.s1, self.scatter_mask.shape())
        self._points = list()
        logger.info("Building Scatter Plot Window; Please wait..")
        for i in range(len(self.img1.flatten())):
            self._points.append(QtCore.QPointF(self.img1.flatten()[i], self.img2.flatten()[i]))

        selected = [roiShape.contains(pt) for pt in self._points]
        img_selected = np.reshape(selected, (self.img1.shape))

        self.masked_img = singleStackViewer(img_selected * self.img1, gradient="bipolar")
        self.masked_img.show()

    def pg_export_correlation(self):

        exporter = pg.exporters.CSVExporter(self.w1)
        exporter.parameters()["columnMode"] = "(x,y,y,y) for all plots"
        file_name = QFileDialog().getSaveFileName(self, "save correlation", "", "spectrum and fit (*csv)")
        if file_name[0]:
            exporter.export(file_name[0] + ".csv")
            self.statusbar.showMessage(f"Data saved to {file_name[0]}")
        else:
            pass

    def tiff_export_images(self):
        file_name = QFileDialog().getSaveFileName(self, "save images", "", "spectrum and fit (*tiff)")
        if file_name[0]:
            tf.imwrite(file_name[0] + ".tiff", np.dstack([self.img1, self.img2]).T)
            self.statusbar.showMessage(f"Images saved to {file_name[0]}")
        else:
            pass

class LoadingScreen(QtWidgets.QSplashScreen):
    def __init__(self):
        super(LoadingScreen, self).__init__()
        uic.loadUi(os.path.join(ui_dir, "animationWindow.ui"), self)
        self.setWindowOpacity(0.65)
        self.movie = QMovie("uis/animation.gif")
        self.label.setMovie(self.movie)

    def mousePressEvent(self, event):
        # disable default "click-to-dismiss" behaviour
        pass

    def startAnimation(self):
        self.movie.start()
        self.show()

    def stopAnimation(self):
        self.movie.stop()
        self.hide()

class CompositeScatterPlot(QtWidgets.QMainWindow):
    def __init__(self, scatterPoints, fitLine, maskImages, fitEquations, nameTuple):
        super(CompositeScatterPlot, self).__init__()

        uic.loadUi(os.path.join(ui_dir, "multipleScatterFit.ui"), self)
        self.user_wd = os.path.abspath("~")
        self.centralwidget.setStyleSheet(open(os.path.join(ui_dir, "css/defaultStyle.css")).read())

        self.scatterPoints = scatterPoints
        # print(f"{np.shape(self.scatterPoints) = }")
        self.fitLine = fitLine
        #print(f"{np.shape(self.fitLine) = }")
        self.scatterColors = ["r", (0, 115, 0), (4, 186, 186), "c", "w", "k"]
        self.fitColors = ["b", "r", "m", "k", "b"]
        self.roiNames = list(fitEquations.keys())
        self.fitEqns = list(fitEquations.values())
        # print(f"{np.shape(self.roiNames) = }")
        # print(f"{np.shape(self.fitEqns) = }")
        self.nameTuple = nameTuple
        self.maskImages = maskImages

        # self.scatterViewer.setBackground('w')
        # set the graphicslayoutwidget in the ui as canvas
        self.canvas = self.scatterViewer.addPlot()
        self.canvas.addLegend()
        self.canvas.setLabel("bottom", self.nameTuple[0], "counts")
        self.canvas.setLabel("left", self.nameTuple[1], "counts")

        # connections
        self.actionExport.triggered.connect(self.exportData)
        self.actionSave_as_PNG.triggered.connect(self.exportAsPNG)
        self.actionGenerate_MultiColor_Mask.triggered.connect(self.generateMultiColorView)
        self.actionWhite.triggered.connect(lambda: self.scatterViewer.setBackground("w"))
        self.actionBlack.triggered.connect(lambda: self.scatterViewer.setBackground("k"))

        with pg.BusyCursor():

            for arr, fitline, clr, fitClr, rname, feqn in zip(
                self.scatterPoints, self.fitLine, self.scatterColors, self.fitColors, self.roiNames, self.fitEqns
            ):

                sctrPoints = []
                for pt in arr:
                    sctrPoints.append(
                        {"pos": (pt[0], pt[1]), "data": "id", "size": 3, "pen": pg.mkPen(None), "brush": clr}
                    )

                # generate a scatter plot item
                self.scattered = pg.ScatterPlotItem(size=4.5, pen=clr, brush=pg.mkBrush(5, 214, 255, 200))
                # set scatter plot data
                self.scattered.addPoints(sctrPoints, name=rname)

                # set z value negative to show scatter data behind the fit line
                self.scattered.setZValue(-10)

                # add scatter plot to the canvas
                self.canvas.addItem(self.scattered)

                # generate plotitem for fit line
                self.fitLinePlot = pg.PlotDataItem(pen=pg.mkPen(fitClr, width=4.5))

                # set line plot data
                self.fitLinePlot.setData(fitline, name=feqn)

                # add line plot to the canvas
                self.canvas.addItem(self.fitLinePlot)

    def generateMultiColorView(self):
        self.multichanneldict = {}

        for n, (colorName, image, rname) in enumerate(zip(cmap_dict.keys(), self.maskImages, self.roiNames)):
            low, high = np.min(image), np.max(image)
            self.multichanneldict[rname] = {
                "ImageName": rname,
                "ImageDir": ".",
                "Image": image,
                "Color": colorName,
                "CmapLimits": (low, high),
                "Opacity": 1.0,
            }

        # print( self.multichanneldict)
        self.muli_color_window = MultiChannelWindow(image_dict=self.multichanneldict)
        self.muli_color_window.show()

    def exportData(self):

        exporter = pg.exporters.CSVExporter(self.canvas)
        # exporter.parameters()['columnMode'] = '(x,y,y,y) for all plots'
        file_name = QFileDialog().getSaveFileName(self, 
                                                  "Save CSV Data", 
                                                  os.path.join(self.user_wd,"scatter.csv"), 
                                                  "image file (*csv)")
        if file_name[0]:
            exporter.export(file_name[0])
            self.statusbar.showMessage(f"Data saved to {file_name[0]}")
            self.user_wd = os.path.dirname(file_name[0])
        else:
            pass

    def exportAsPNG(self):
        file_name = QFileDialog().getSaveFileName(self, 
                                                 "Save Image", 
                                                 os.path.join(self.user_wd,"image.png"),
                                                 "PNG(*.png);; TIFF(*.tiff);; JPG(*.jpg)"
        )
        exporter = pg.exporters.ImageExporter(self.canvas)

        if file_name[0]:
            exporter.export(file_name[0])
            self.statusbar.showMessage(f"Image saved to {file_name[0]}")
            self.user_wd = os.path.dirname(file_name[0])
        else:
            pass


class StackInfo(QtWidgets.QMainWindow):
    def __init__(self, text_to_write: str = " "):
        super(StackInfo, self).__init__()
        uic.loadUi(os.path.join(ui_dir, "log.ui"), self)
        self.user_wd = os.path.abspath("~")

        self.text_to_write = text_to_write
        self.pte_run_cmd.setPlainText(self.text_to_write)

        # connections
        self.pb_save_cmd.clicked.connect(self.save_file)
        self.pb_clear_cmd.clicked.connect(self.clear_text)

    def save_file(self):
        S__File = QFileDialog.getSaveFileName(None, "SaveFile", "/", "txt Files (*.txt)")

        Text = self.pte_run_cmd.toPlainText()
        if S__File[0]:
            with open(S__File[0], "w") as file:
                file.write(Text)

    def clear_text(self):
        self.pte_run_cmd.clear()


def start_xmidas():
    def formatter(prog):
        # Set maximum width such that printed help mostly fits in the RTD theme code block (documentation).
        return argparse.RawDescriptionHelpFormatter(prog, max_help_position=20, width=90)
    
    parser = argparse.ArgumentParser(
        description=f"XMidas: v{__version__}",
        formatter_class=formatter,
    )
    parser.parse_args()
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s : %(levelname)s : %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(stream_handler)
    if version.parse(PYQT_VERSION_STR) >= version.parse("5.14"):
        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )


    app = QtWidgets.QApplication(sys.argv)
    # app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    window = midasWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    start_xmidas()

__version__ = "1.0.0"  # or whatever version you want
