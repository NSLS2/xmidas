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
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QApplication, QAbstractItemView, QTableWidgetItem
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool, PYQT_VERSION_STR

from xmidas.utils import *
from xmidas.utils.color_maps import *
from xmidas.models.encoders import jsonEncoder
from xmidas.utils.utils import xanes_fitting, xanes_fitting_1D, xanes_fitting_Binned, interploate_E, get_sum_spectra, get_mean_spectra, normalize_and_scale
from xmidas.gui.windows.multichannel_viewer import MultiChannelWindow
from xmidas.gui.windows.decomposition_viewer import *

cmap_dict = create_color_maps()
ui_dir = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../layout"
))

class XANESViewer(QtWidgets.QMainWindow):
    def __init__(self, im_stack=None, e_list=None, refs=None, ref_names=None):
        super(XANESViewer, self).__init__()

        uic.loadUi(os.path.join(ui_dir, "XANESViewer.ui"), self)
        self.user_wd = os.path.abspath("~")
        self.centralwidget.setStyleSheet(open(os.path.join(ui_dir, "css/defaultStyle.css")).read())

        self.im_stack = im_stack
        self.e_list = e_list
        self.refs = refs
        self.ref_names = ref_names
        self.selected = self.ref_names
        self.fitResultDict = {}
        self.fit_method = self.cb_xanes_fit_model.currentText()
        self.alphaForLM = self.dsb_alphaForLM.value()
        self.xpixel = None
        self.ypixel = None
        self.xdata_eshifted= self.e_list + self.sb_e_shift.value()
        self.decon_ims, self.rfactor, self.coeffs_arr = xanes_fitting(
                                                                      self.im_stack, 
                                                                      self.xdata_eshifted, 
                                                                      self.refs, 
                                                                      method=self.fit_method, 
                                                                      alphaForLM=self.alphaForLM)
        
        self.add_roi()
        self.add_point_item()
        self.scrollBar_setup()
        self.display_image_data()
        self.display_references()
        self.fit_roi_spectrum()

        # connections
        self.sb_e_shift.valueChanged.connect(self.fit_roi_spectrum)
        self.pb_re_fit.clicked.connect(self.re_fit_xanes)
        self.pb_edit_refs.clicked.connect(self.choose_refs)
        self.image_roi.sigRegionChanged.connect(self.fit_roi_spectrum)
        self.hsb_xanes_stk.valueChanged.connect(self.display_image_data)
        self.hsb_chem_map.valueChanged.connect(self.display_image_data)
        # self.pb_showMultiColor.clicked.connect(self.generateMultiColorView)
        # self.pb_showCompSpec.clicked.connect(self.showComponentXANES)
        self.pb_showCompSpec.clicked.connect(self.generateCompoisteImageSpectrumView)
        self.image_view.mousePressEvent = self.fit_point_spectrum
        #self.image_view_maps.mousePressEvent = self.fit_point_spectrum

        # menu
        self.actionSave_Chem_Map.triggered.connect(self.save_chem_map)
        self.actionSave_R_factor_Image.triggered.connect(self.save_rfactor_img)
        #self.actionSave_Live_Fit_Data.triggered.connect(self.pg_export_spec_fit)
        self.actionSave_Live_Fit_Data.triggered.connect(lambda:self.export_data_and_params(folder=None))
        self.actionExport_Fit_Stats.triggered.connect(self.exportFitResults)
        self.actionExport_Ref_Plot.triggered.connect(self.pg_export_references)
        self.action_export_sum_fit_data.triggered.connect(lambda:self.plot_sum_spectrum_and_save(save_to=None))
        self.pb_open_mask_maker.clicked.connect(self.create_masked_xanes)
        self.pb_open_cluster_mask.clicked.connect(self.create_decmpose_mask)
        self.pb_norm_chem_map.clicked.connect(self.normalize_chem_map)
        self.pb_cluster_chem_map.clicked.connect(self.cluster_chem_map)

    def add_roi(self):
        (self.dim1, self.dim2, self.dim3) = self.im_stack.shape
        self.cn = int(self.dim2 // 2)
        self.sz = np.max([int(self.dim2 * 0.15), int(self.dim3 * 0.15)])
        self.stack_center = int(self.dim1 // 2)
        self.image_roi = pg.RectROI(
            [int(self.dim3 // 2), int(self.dim2 // 2)],
            [self.sz, self.sz],
            pen="w",
            maxBounds=QtCore.QRectF(0, 0, self.dim3, self.dim2),
        )
        self.image_roi.addTranslateHandle([0, 0], [2, 2])
        self.image_roi.addRotateHandle([0, 1], [2, 2])
        try:
            self.image_view.removeItem(self.image_roi)
        except: pass

        self.image_view.addItem(self.image_roi)

    def add_point_item(self):
        self.clicked_point = pg.ScatterPlotItem(
                                                size=9, 
                                                pen=pg.mkPen(None), 
                                                brush=pg.mkBrush(255, 255, 255),
                                                hoverable=True,
                                                hoverBrush=pg.mkBrush(255, 16, 240)
                                                )
        self.clicked_point_for_map = pg.ScatterPlotItem(
                                                size=9, 
                                                pen=pg.mkPen(None), 
                                                brush=pg.mkBrush(255, 255, 255),
                                                hoverable=True,
                                                hoverBrush=pg.mkBrush(255, 16, 240)
                                                )
        try:
            self.image_view.removeItem(self.clicked_point)
            self.image_view_maps.removeItem(self.clicked_point_for_map)
        except: pass

        self.image_view.addItem(self.clicked_point)
        self.image_view_maps.addItem(self.clicked_point_for_map)
        # Ensure scatterPlotItem is always at top
        self.clicked_point.setZValue(2) 


    def scrollBar_setup(self):
        self.hsb_xanes_stk.setValue(self.stack_center)
        self.hsb_xanes_stk.setMaximum(self.dim1 - 1)
        self.hsb_chem_map.setValue(0)
        self.hsb_chem_map.setMaximum(self.decon_ims.shape[-1] - 1)

    
    def import_stack(self):
        filename = QFileDialog().getOpenFileName(
            self, "Select image data", self.user_wd, "image file( *tiff *tif )"
        )
        self.file_name = str(filename[0])
        self.user_wd = os.path.dirname(self.file_name)

        if filename[0]:
            self.im_stack = tf.imread(filename[0]) 

        #TODO import energy, look for default files first, if not open file dialogue,
        #raise cannot proceed
            
        # if str(self.efilePath).endswith("log_tiff.txt"):
        #         self.energy = energy_from_logfile(logfile=str(self.efilePath))
        #         logger.info("Log file from pyxrf processing")

        # else:
        #     self.energy = np.loadtxt(str(self.efilePath))
        #     self.change_color_on_load(self.pb_elist_xanes)
        #     logger.info("Energy file loaded")

        self.decon_ims, self.rfactor, self.coeffs_arr = xanes_fitting(self.im_stack, 
                                                                      self.xdata_eshifted, 
                                                                      self.refs, 
                                                                      method=self.fit_method,
                                                                      alphaForLM=self.alphaForLM
                                                                      )

        self.add_roi()
        self.add_point_item()
        self.scrollBar_setup()
        self.display_image_data()
        self.display_references()
        self.fit_roi_spectrum()

    def display_image_data(self):

        self.image_view.setImage(self.im_stack[self.hsb_xanes_stk.value()])
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.setPredefinedGradient("viridis")

        self.image_view_maps.setImage(self.decon_ims.transpose(2, 0, 1)[self.hsb_chem_map.value()])
        self.image_view_maps.setPredefinedGradient("bipolar")
        self.image_view_maps.ui.menuBtn.hide()
        self.image_view_maps.ui.roiBtn.hide()

    def display_references(self):

        self.inter_ref = interploate_E(self.refs, self.xdata_eshifted)
        self.plt_colors = ["c", "m", "y", "w"] * 10
        self.spectrum_view_refs.addLegend()
        for ii in range(self.inter_ref.shape[0]):
            if len(self.selected) != 0:
                self.spectrum_view_refs.plot(
                    self.xdata_eshifted,
                    self.inter_ref[ii],
                    pen=pg.mkPen(self.plt_colors[ii], width=2),
                    name=self.selected[1:][ii],
                )
            else:
                self.spectrum_view_refs.plot(
                    self.xdata_eshifted,
                    self.inter_ref[ii],
                    pen=pg.mkPen(self.plt_colors[ii], width=2),
                    name="ref" + str(ii + 1),
                )

    def choose_refs(self):
        "Interactively exclude some standards from the reference file"
        self.ref_edit_window = RefChooser(
            self.ref_names,
            self.im_stack,
            self.e_list,
            self.refs,
            self.sb_e_shift.value(),
            self.cb_xanes_fit_model.currentText(),
        )
        self.ref_edit_window.show()
        # self.rf_plot = pg.plot(title="RFactor Tracker")

        # connections
        self.ref_edit_window.choosenRefsSignal.connect(self.update_refs)
        self.ref_edit_window.fitResultsSignal.connect(self.plotFitResults)


    def create_masked_xanes(self):
        "create mask and apply to xanes 3d array"

        self.mask_creator = MaskMaker(self.im_stack,self.decon_ims.transpose(2, 0, 1)[self.hsb_chem_map.value()])
        self.mask_creator.show()

        #connections
        self.mask_creator.mask_signal.connect(self.apply_mask_to_xanes)

    def create_decmpose_mask(self):

        self.decomposer = DecomposeViewer(self.im_stack,None,self.xdata_eshifted)
        self.decomposer.show()
        #connections
        self.decomposer.mask_signal.connect(self.apply_mask_to_xanes)
        self.decomposer.mask_and_path_signal.connect(self.recieve_mask_and_save)

    def cluster_chem_map(self):

        self.decomposer2 = DecomposeViewer(self.decon_ims.transpose(2, 0, 1)[self.hsb_chem_map.value()],
                                           self.im_stack,
                                           self.xdata_eshifted)
        

        # self.decomposer2 = DecomposeViewer(self.decon_ims.transpose(2, 0, 1),
        #                                    self.im_stack,
        #                                    self.xdata_eshifted)
        self.decomposer2.show()
        #connections
        self.decomposer2.mask_signal.connect(self.apply_mask_to_xanes)
        self.decomposer2.mask_and_path_signal.connect(self.recieve_mask_and_save)

    def plot_sum_spectrum(self):
        self.xdata_eshifted = self.e_list + self.sb_e_shift.value()
        self.ydata1 = get_sum_spectra(self.im_stack)
        self.plot_data_and_fit()
        self.re_fit_xanes()


    def plot_sum_spectrum_and_save(self, save_to = None):
        self.xdata_eshifted = self.e_list + self.sb_e_shift.value()
        self.ydata1 = get_sum_spectra(self.im_stack)
        self.plot_data_and_fit(plot_name="Sum Spectrum")
        self.re_fit_xanes()
        self.export_data_and_params(folder=save_to)

    def recieve_mask_and_save(self,signals):
        self.apply_mask_to_xanes(signals[0])
        QtTest.QTest.qWait(5000)
        self.plot_sum_spectrum_and_save(save_to = signals[1])
        QtTest.QTest.qWait(2000)


    def normalize_chem_map(self):

        stack_ = normalize_and_scale(self.decon_ims.transpose(2, 0, 1))
        self.decon_ims = stack_.transpose(1, 2, 0)
        self.display_image_data()


    def apply_mask_to_xanes(self, masked_array):

        """ Apply the recieved mask and plot mean spectrum and fit """

        #self.im_stack = self.im_stack*mask[np.newaxis,:,:]
        self.im_stack = masked_array
        self.display_image_data()
        self.xdata_eshifted = self.e_list + self.sb_e_shift.value()
        self.ydata1 = get_mean_spectra(self.im_stack)
        self.plot_data_and_fit()
        self.re_fit_xanes()

    def update_refs(self, list_):
        self.selected = list_  # list_ is the signal from ref chooser
        self.fit_roi_spectrum()
        self.re_fit_xanes()

    def fit_point_spectrum(self, event):
        if event.type() == QtCore.QEvent.Type.MouseButtonDblClick:
            if event.button() == QtCore.Qt.LeftButton:
                self.xpixel = int(self.image_view.view.mapSceneToView(event.pos()).x())
                zlim, ylim, xlim = self.im_stack.shape

                if self.xpixel > xlim:
                    self.xpixel = xlim

                self.ypixel = int(self.image_view.view.mapSceneToView(event.pos()).y())
                if self.ypixel > ylim:
                    self.ypixel = ylim
                
                self.clicked_point.setData(pos=[[self.xpixel+0.5,self.ypixel+0.5],])
                self.clicked_point_for_map.setData(pos=[[self.xpixel+0.5,self.ypixel+0.5],])
                self.xdata_eshifted = self.e_list + self.sb_e_shift.value()
                self.ydata1 = self.im_stack[:, self.ypixel, self.xpixel]

                self.plot_data_and_fit(plot_name=f"Point Spectrum; x= {self.xpixel}, y= {self.ypixel}")
    
    def plot_data_and_fit(self, plot_name = "Data"):

        if self.cb_normalize_before_fit.isChecked():
            self.ydata1 =self.ydata1/self.ydata1[-1]

        self.fit_method = self.cb_xanes_fit_model.currentText()
        self.spectrum_view.addLegend()
        if len(self.selected) != 0:

            self.inter_ref = interploate_E(self.refs[self.selected], self.xdata_eshifted)
            stats, coeffs = xanes_fitting_1D(
                self.ydata1,
                self.xdata_eshifted,
                self.refs[self.selected],
                method=self.fit_method,
                alphaForLM=self.alphaForLM,
            )

        else:
            self.inter_ref = interploate_E(self.refs, self.xdata_eshifted)
            stats, coeffs = xanes_fitting_1D(
                self.ydata1, self.xdata_eshifted, self.refs, method=self.fit_method, alphaForLM=self.alphaForLM
            )

        self.fit_ = np.dot(coeffs, self.inter_ref)
        pen = pg.mkPen("b", width=1.5)
        pen2 = pg.mkPen("r", width=1.5)
        # pen3 = pg.mkPen("y", width=1.5)
        self.spectrum_view.addLegend()
        self.spectrum_view.setLabel("bottom", "Energy")
        self.spectrum_view.setLabel("left", "Intensity", "A.U.")
        self.spectrum_view.plot(self.xdata_eshifted, 
                                self.ydata1, 
                                pen=pen, 
                                clear=True,
                                symbol="o",
                                symbolSize=6,
                                symbolBrush="b",
                                name=f"{plot_name}")
        
        self.spectrum_view.plot(self.xdata_eshifted, self.fit_, name="Fit", pen=pen2)

        for n, (coff, ref, plt_clr) in enumerate(zip(coeffs, self.inter_ref, self.plt_colors)):

            if len(self.selected) != 0:

                self.spectrum_view.plot(self.xdata_eshifted, np.dot(coff, ref), name=self.selected[1:][n], pen=plt_clr)
            else:
                self.spectrum_view.plot(self.xdata_eshifted, np.dot(coff, ref), name="ref" + str(n + 1), pen=plt_clr)
        # set the rfactor value to the line edit slot
        self.results = {
            "ref_stds":self.selected[1:],
            "Coefficients":coeffs, 
            "R-Factor": stats['R_Factor'], 
            "R-Square": stats['R_Square'],
            "Chi-Square": stats['Chi_Square'],
            "Reduced Chi-Square": stats['Reduced Chi_Square']}

        self.fit_results.setText("\n".join(f"{key}: {value}" for key, value in self.results.items()))

    def fit_roi_spectrum(self):

        self.roi_img = self.image_roi.getArrayRegion(self.im_stack, self.image_view.imageItem, axes=(1, 2))
        sizex, sizey = self.roi_img.shape[1], self.roi_img.shape[2]
        posx, posy = self.image_roi.pos()
        self.roi_info.setText(f"ROI_Pos: {int(posx)},{int(posy)} ROI_Size: {sizex},{sizey}")

        self.xdata_eshifted = self.e_list + self.sb_e_shift.value()
        self.ydata1 = get_mean_spectra(self.roi_img)

        self.plot_data_and_fit(plot_name="ROI Spectrum")        

    def re_fit_xanes(self):
        self.fit_params = {"e_shit":self.sb_e_shift.value(),
                           "method":self.cb_xanes_fit_model.currentText(),
                           "regularization":self.dsb_alphaForLM.value()}
        

        if len(self.selected) != 0:
            self.decon_ims, self.rfactor, self.coeffs_arr = xanes_fitting(
                self.im_stack,
                self.e_list + self.sb_e_shift.value(),
                self.refs[self.selected],
                method=self.cb_xanes_fit_model.currentText(),
                alphaForLM=self.alphaForLM,
            )
        else:
            # if non athena file with no header is loaded no ref file cannot be edited
            self.decon_ims, self.rfactor, self.coeffs_arr = xanes_fitting(
                self.im_stack,
                self.e_list + self.sb_e_shift.value(),
                self.refs,
                method=self.cb_xanes_fit_model.currentText(),
                alphaForLM=self.alphaForLM,
            )

        # rfactor is a list of all spectra so take the mean
        self.rfactor_mean = np.mean(self.rfactor)
        self.image_view_maps.setImage(self.decon_ims.transpose(2, 0, 1))
        self.scrollBar_setup()

    def plotFitResults(self, decon_ims, rfactor_mean, coeff_array):
        # upadte the chem maps and scrollbar params
        self.image_view_maps.setImage(decon_ims.transpose(2, 0, 1))
        self.hsb_chem_map.setValue(0)
        self.hsb_chem_map.setMaximum(decon_ims.shape[-1]-1)

        # set the rfactor value to the line edit slot
        self.le_r_sq.setText(f"{rfactor_mean :.4f}")

    def showComponentXANES(self):
        compNum = self.hsb_chem_map.value()
        currentComp = self.decon_ims.transpose(2, 0, 1)[compNum]
        currentCompMask = currentComp > 0
        yData = applyMaskGetMeanSpectrum(self.im_stack, currentCompMask)
        xanes_comp_plot = pg.plot(
            self.e_list + self.sb_e_shift.value(),
            yData,
            title=f"Component_{compNum}",
            pen=pg.mkPen("y", width=2, style=QtCore.Qt.DotLine),
            symbol="o",
        )
        xanes_comp_plot.setLabel("bottom", "Energy (keV)")
        xanes_comp_plot.setLabel("left", "Intensity")

    def plotDeconvSpectrum(self, clusterSigma=0):

        try:
            self.ref_plot.close()

        except:
            pass

        self.ref_plot = plot(title="Deconvoluted XANES Spectra")
        self.ref_plot.setLabel("bottom", "Energy")
        self.ref_plot.setLabel("left", "Intensity")
        self.ref_plot.addLegend()

        for n, compImage in enumerate(self.decon_ims.transpose(2, 0, 1)):
            mask = np.where(compImage > clusterSigma * np.std(compImage), compImage, 0)

            self.ref_plot.plot(
                self.xdata_eshifted,
                get_mean_spectra(self.im_stack * mask),
                pen=pg.mkPen(self.plt_colors[n], width=2),
                name=f'Component_{n + 1}'
            )

    def generateCompoisteImageSpectrumView(self):
        self.multichanneldict = {}

        spectrumDF = getDeconvolutedXANESSpectrum(self.im_stack, self.decon_ims.transpose(2, 0, 1),
                                                  self.xdata_eshifted, clusterSigma=0)

        for n, (colorName, image) in enumerate(zip(cmap_dict.keys(), self.decon_ims.transpose((2, 0, 1)))):
            low, high = np.min(image), np.max(image)
            self.multichanneldict[f'Image {n + 1}'] = {'ImageName': f'Image {n + 1}',
                                                       'ImageDir': '.',
                                                       'Image': image,
                                                       'Color': colorName,
                                                       'CmapLimits': (low, high),
                                                       'Opacity': 1.0
                                                       }
        self.muli_color_window = MultiXANESWindow(image_dict=self.multichanneldict, spec_df=spectrumDF)
        self.muli_color_window.show()

    def generateMultiColorView(self):
        self.multichanneldict = {}

        for n, (colorName, image) in enumerate(zip(cmap_dict.keys(), self.decon_ims.transpose((2, 0, 1)))):
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

    def save_chem_map(self):
        file_name = QFileDialog().getSaveFileName(self,
                                                   "save image",
                                                   os.path.join(self.user_wd,"chemical_map.tiff"),
                                                    "image data (*tiff)")
        if file_name[0]:
            tf.imwrite(file_name[0], np.float32(self.decon_ims.transpose(2, 0, 1)))
            self.user_wd = os.path.dirname(file_name[0])
        else:
            logger.error("No file to save")
            pass

    def save_rfactor_img(self):
        file_name = QFileDialog().getSaveFileName(self, 
                                                  "save image", 
                                                  os.path.join(self.user_wd,"r-factor_map.tiff"), 
                                                  "image data (*tiff)")
        if file_name[0]:
            tf.imwrite(file_name[0], np.float32(self.rfactor), imagej=True)
            self.user_wd = os.path.dirname(file_name[0])
        else:
            logger.error("No file to save")
            pass

    def save_spec_fit(self):
        try:
            to_save = np.column_stack([self.xdata_eshifted, self.ydata1, self.fit_])
            file_name = QFileDialog().getSaveFileName(self, 
                                                      "save spectrum",
                                                        os.path.join(self.user_wd,"spec_fit.txt"), 
                                                        "spectrum and fit (*txt)")
            if file_name[0]:
                np.savetxt(file_name[0] + ".txt", to_save)
                self.user_wd = os.path.dirname(file_name[0])
            else:
                pass
        except Exception:
            logger.error("No file to save")
            pass

    def pg_export_spec_fit(self):

        exporter = pg.exporters.CSVExporter(self.spectrum_view.plotItem)
        exporter.parameters()["columnMode"] = "(x,y,y,y) for all plots"
        file_name = QFileDialog().getSaveFileName(self, 
                                                  "save spectrum", 
                                                  os.path.join(self.user_wd,"spec_fit.txt"), 
                                                  "spectrum and fit (*csv)")
        if file_name[0]:
            exporter.export(file_name[0] + ".csv")
            self.user_wd = os.path.dirname(file_name[0])
        else:
            pass

    def pg_export_references(self):

        exporter = pg.exporters.CSVExporter(self.spectrum_view_refs.plotItem)
        exporter.parameters()["columnMode"] = "(x,y,y,y) for all plots"
        file_name = QFileDialog().getSaveFileName(self,
                                                   "save references", 
                                                   os.path.join(self.user_wd,"xanes_references.csv"), 
                                                   "column data (*csv)"
        )
        if file_name[0]:
            exporter.export(file_name[0])
            self.user_wd = os.path.dirname(file_name[0])
        else:
            pass

    def exportFitResults(self):
        file_name = QFileDialog().getSaveFileName(self, 
                                                  "save txt",
                                                  os.path.join(self.user_wd,"xanes_1D_fit_results.json"),
                                                  "txt data (*txt)")
        if file_name[0]:
            with open(file_name[0], "w") as file:
                json.dump(self.results, file, indent=4)
            self.user_wd = os.path.dirname(file_name[0])
        else:
            pass

    def export_data_and_params(self, folder = None):

        if folder is None:
            file_name,_ = QFileDialog().getSaveFileName(self, 
                                                    "save to folder",
                                                    os.path.join(self.user_wd, "xanes_data"),
                                                    options = QFileDialog.ShowDirsOnly)
            
        else:
            file_name = folder
        exporter_csv = pg.exporters.CSVExporter(self.spectrum_view.plotItem)
        exporter_csv.parameters()["columnMode"] = '(x,y) per plot'
        exporter_csv.parameters()["separator"] = 'comma'

        exporter_png = pg.exporters.ImageExporter(self.spectrum_view.getViewBox())
        exporter_img_png = pg.exporters.ImageExporter(self.image_view.getView())

        exporter_svg = pg.exporters.SVGExporter(self.spectrum_view.getViewBox())
        exporter_img_svg = pg.exporters.SVGExporter(self.image_view.getView())

        if file_name:
            os.makedirs(file_name[0], exist_ok=True)
            os.makedirs(os.path.join(file_name+"/csv_files"), exist_ok=True)
            os.makedirs(os.path.join(file_name+"/png_files"), exist_ok=True)
            os.makedirs(os.path.join(file_name+"/svg_files"), exist_ok=True)
            os.makedirs(os.path.join(file_name+"/tiff_files"), exist_ok=True)
            exporter_csv.export(os.path.join(file_name+"/csv_files/xanes_fit.csv"))
            exporter_png.export(os.path.join(file_name+"/png_files/xanes_fit_image.png"))
            exporter_img_png.export(os.path.join(file_name+"/png_files/stack_image.png"))
            exporter_svg.export(os.path.join(file_name+"/svg_files/xanes_fit.svg"))
            exporter_img_svg.export(os.path.join(file_name+"/svg_files/stack_image.svg"))

            tf.imwrite(os.path.join(file_name+"/tiff_files/chem_map.tiff"), np.float32(self.decon_ims.transpose(2, 0, 1)))
            tf.imwrite(os.path.join(file_name+"/tiff_files/rfactor.tiff"), np.float32(self.rfactor), imagej=True)
            tf.imwrite(os.path.join(file_name+"/tiff_files/stack_image.tiff"), np.float32(self.im_stack[self.hsb_xanes_stk.value()]))

            roi_and_pixel_params = {"sizex":self.roi_img.shape[1],"sizey":self.roi_img.shape[2],
                                    "posx":self.image_roi.pos()[0],"posy":self.image_roi.pos()[1],
                                    "pixelx":self.xpixel, "pixely":self.ypixel}
            
            self.fit_params = {"e_shit":self.sb_e_shift.value(),
                           "method":self.cb_xanes_fit_model.currentText(),
                           "regularization":self.dsb_alphaForLM.value()}
            export_params = {"fit_params":self.fit_params,
                             "roi_or_pixel_params":roi_and_pixel_params,
                             "fit_results":self.results}
            
            with open(os.path.join(file_name+"/fit_params.json"), "w") as fp:
                json.dump(export_params, fp, indent=4, cls = jsonEncoder)
            self.user_wd = os.path.dirname(file_name)
            QMessageBox.information(self,"Saved", f"data saved to {self.user_wd}")


class RefChooser(QtWidgets.QMainWindow):
    choosenRefsSignal = QtCore.pyqtSignal(list)
    fitResultsSignal = QtCore.pyqtSignal(np.ndarray, float, np.ndarray)

    def __init__(self, ref_names, im_stack, e_list, refs, e_shift, fit_model):
        super().__init__()
        uic.loadUi(os.path.join(ui_dir, "RefChooser.ui"), self)

        self.user_wd = os.path.abspath("~")
        self.setStyleSheet(open(os.path.join(ui_dir, "css/defaultStyle.css")).read())

        self.ref_names = ref_names
        self.refs = refs
        self.im_stack = im_stack
        self.e_list = e_list
        self.e_shift = e_shift
        self.fit_model = fit_model

        self.all_boxes = []
        self.rFactorList = []

        self.setupUI()

    def setupUI(self):
        self.displayCombinations()

        self.tableWidget.setStyleSheet(
            "background-color: white; selection-background-color: rgb(200,0,0);"
        )

        self.selectionLine = pg.InfiniteLine(
            pos=1, angle=90, pen=pg.mkPen("m", width=2.5), movable=True, label="Move Me!"
        )
        self.stat_view.setLabel("bottom", "Fit ID")
        self.stat_view.setLabel("left", "Reduced Chi^2")

        for n, name in enumerate(self.ref_names):
            checkbox = QtWidgets.QCheckBox(self.ref_box_frame)
            checkbox.setObjectName(name)
            checkbox.setText(name)
            checkbox.setChecked(n == 0)
            checkbox.setEnabled(n != 0)
            checkbox.toggled.connect(self.enableApply)
            self.gridLayout_2.addWidget(checkbox, n, 0, 1, 1)
            self.all_boxes.append(checkbox)

        self.pb_apply.clicked.connect(self.clickedWhichAre)
        self.pb_combo.clicked.connect(self.tryAllCombo)
        self.actionExport_Results_csv.triggered.connect(self.exportFitResults)
        self.selectionLine.sigPositionChanged.connect(self.updateFitWithLine)
        self.tableWidget.itemSelectionChanged.connect(self.updateWithTableSelection)
        self.stat_view.mouseDoubleClickEvent = self.moveSelectionLine
        self.sb_max_combo.valueChanged.connect(self.displayCombinations)
        self.pb_sort_with_r.clicked.connect(self.sortTable)
        self.cb_sorter.currentTextChanged.connect(self.sortTable)

    def populateChecked(self):
        self.onlyCheckedBoxes = [cb.objectName() for cb in self.all_boxes if cb.isChecked()]

    @QtCore.pyqtSlot()
    def clickedWhichAre(self):
        self.populateChecked()
        self.choosenRefsSignal.emit(self.onlyCheckedBoxes)

    def generateRefList(self, ref_list, maxCombo, minCombo=1):
        if maxCombo > len(ref_list):
            raise ValueError("Maximum combinations cannot exceed number of references")
        combinations_list = [combo for i in range(minCombo, maxCombo + 1) for combo in combinations(ref_list, i)]
        return len(combinations_list), combinations_list

    def displayCombinations(self):
        niter, self.iter_list = self.generateRefList(self.ref_names[1:], self.sb_max_combo.value())
        self.label_nComb.setText(f"{niter} Combinations")

    @QtCore.pyqtSlot()
    def tryAllCombo(self):
        self.rfactor_list = []
        self.results_data = []

        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(9)
        self.tableWidget.setHorizontalHeaderLabels([
            "Fit Number", "References", "Coefficients", "Sum of Coefficients",
            "R-Factor", "R^2", "chi^2", "red-chi^2", "Score"
        ])

        niter, self.iter_list = self.generateRefList(self.ref_names[1:], self.sb_max_combo.value())

        for n, refs in enumerate(self.iter_list):
            self.statusbar.showMessage(f"{n + 1}/{niter}")
            selectedRefs = [self.ref_names[0]] + list(refs)
            self.fit_combo_progress.setValue(int((n + 1) * 100 / niter))

            self.stat, self.coeffs_arr = xanes_fitting_Binned(
                self.im_stack, self.e_list + self.e_shift, self.refs[selectedRefs], method=self.fit_model
            )

            self.rfactor_list.append(self.stat["Reduced Chi_Square"])
            self.stat_view.plot(
                x=np.arange(n + 1), y=self.rfactor_list, clear=True,
                title="Reduced Chi^2", pen=pg.mkPen("y", width=2, style=QtCore.Qt.PenStyle.DotLine), symbol="o"
            )

            fit_score = (self.stat["R_Square"] + np.sum(self.coeffs_arr)) / (
                self.stat["R_Factor"] + self.stat["Reduced Chi_Square"])

            resultsDict = {
                "Fit Number": n,
                "References": str(selectedRefs[1:]),
                "Coefficients": str(np.around(self.coeffs_arr, 4)),
                "Sum of Coefficients": str(np.around(np.sum(self.coeffs_arr), 4)),
                "R-Factor": self.stat["R_Factor"],
                "R^2": self.stat["R_Square"],
                "chi^2": self.stat["Chi_Square"],
                "red-chi^2": self.stat["Reduced Chi_Square"],
                "Score": np.around(fit_score, 4),
            }

            self.results_data.append(resultsDict)
            self.appendToTableWidget(resultsDict)
            QtTest.QTest.qWait(100)

        self.stat_view.addItem(self.selectionLine)

    def appendToTableWidget(self, row_dict):
        row_position = self.tableWidget.rowCount()
        self.tableWidget.insertRow(row_position)
        for col, key in enumerate([
            "Fit Number", "References", "Coefficients", "Sum of Coefficients",
            "R-Factor", "R^2", "chi^2", "red-chi^2", "Score"
        ]):
            item = QtWidgets.QTableWidgetItem(str(row_dict[key]))
            self.tableWidget.setItem(row_position, col, item)

    def exportFitResults(self):
        if not hasattr(self, "results_data"):
            return
        df = pd.DataFrame(self.results_data)
        file_name, _ = QFileDialog.getSaveFileName(self, "Save CSV", "xanes_fit_results_log.csv", "CSV Files (*.csv)")
        if file_name:
            df.to_csv(file_name, index=False)

    def selectTableAndCheckBox(self, x):
        nSelection = int(round(x))
        self.tableWidget.selectRow(nSelection)
        refs_selected = self.iter_list[int(self.tableWidget.item(nSelection, 0).text())]

        for cb in self.findChildren(QtWidgets.QCheckBox):
            if cb.isEnabled():
                cb.setChecked(False)

        for name in refs_selected:
            cb = self.findChild(QtWidgets.QCheckBox, name=name)
            if cb:
                cb.setChecked(True)

    def updateFitWithLine(self):
        pos_x, _ = self.selectionLine.pos()
        x = int(round(pos_x))
        self.selectTableAndCheckBox(x)

    def updateWithTableSelection(self):
        self.selectTableAndCheckBox(self.tableWidget.currentRow())

    def moveSelectionLine(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = self.stat_view.plotItem.vb.mapSceneToView(event.pos())
            self.selectionLine.setPos(pos.x())

    def sortTable(self):
        key_map = {
            "R-Factor": 4,
            "R-Square": 5,
            "Chi-Square": 6,
            "Reduced Chi-Square": 7,
            "Fit Number": 0,
        }
        col_index = key_map.get(self.cb_sorter.currentText(), 0)
        self.tableWidget.sortItems(col_index, QtCore.Qt.SortOrder.AscendingOrder)

    def enableApply(self):
        self.populateChecked()
        self.pb_apply.setEnabled(len(self.onlyCheckedBoxes) > 1)

class MultiXANESWindow(MultiChannelWindow):

    def __init__(self, image_dict=None, spec_df=None, references = None, ):
        super().__init__(image_dict=None)

        self.image_dict = image_dict
        self.spec_df = spec_df

        uic.loadUi(os.path.join(ui_dir, 'MultiImageSpectrumView.ui'), self)
        self.user_wd = os.path.abspath("~")
        # Copy from MultiChannelWindow Start here
        self.canvas = self.img_view.addPlot(title="")
        self.canvas.getViewBox().invertY(True)
        #self.canvas.setZValue(-10)
        self.canvas.setAspectLocked(True)
        self.cb_choose_color.addItems([i for i in cmap_dict.keys()])
        #self.canvas.getViewBox().setBackgroundColor(pg.mkColor(222,222,222))
        #self.canvas.getViewBox().setOpacity(0.5)

        self.image_dict = image_dict
        self.buildFromDictionary()

        self.actionLoad.triggered.connect(self.createMuliColorAndList)
        self.actionLoad_Stack.triggered.connect(self.createMuliColorAndList)
        self.cb_choose_color.currentTextChanged.connect(self.updateImageDictionary)
        self.pb_update_low_high.clicked.connect(self.updateImageDictionary)
        self.listWidget.itemClicked.connect(self.editImageProperties)
        self.listWidget.itemDoubleClicked.connect(self.showOneImageOnly)
        self.pb_show_selected.clicked.connect(self.showOneImageOnly)
        self.pb_show_all.clicked.connect(self.showAllItems)
        self.actionLoad_State_File.triggered.connect(self.importState)
        self.actionSave_State.triggered.connect(self.exportState)
        self.actionSave_View.triggered.connect(self.saveImage)
        # Copy from MultiChannelWindow End here
        self.actionSave_Spectrum_Data.triggered.connect(self.exportDisplayedSpectra)
        self.listWidget_Spectrum.itemClicked.connect(self.plotNormSpectrum)
        self.pb_apply_xanes_norm.clicked.connect(lambda: self.updateSpecData(plotNorm=True))
        self.createMultiSpectrumLibrary()

        [dsb.valueChanged.connect(lambda: self.updateSpecData()) for dsb in
         [self.dsb_norm_Eo, self.dsb_norm_pre1, self.dsb_norm_pre2, self.dsb_norm_post1,
          self.dsb_norm_post2, self.sb_norm_order]]

    def createSpectrumPropertyDict(self, specName, xdata, ydata, e0, pre1, pre2, norm1, norm2, normOrder):
        SingleSpecProperty = {'Name': specName,
                              'Data': (xdata, ydata),
                              'NormParams': [e0, pre1, pre2, norm1, norm2, normOrder]}

        return SingleSpecProperty

    def createMultiSpectrumLibrary(self):
        self.spec_dict = {}
        column_names = self.spec_df.columns
        spec_array = self.spec_df.to_numpy()
        energy = spec_array[:, 0]
        for i in range(self.spec_df.shape[1]):
            if i != 0:
                specData = spec_array[:, i]
                e0_init = energy[np.argmax(np.gradient(specData))]

                pre1, pre2, post1, post2 = xanesNormalization(
                    energy,
                    specData,
                    e0=e0_init,
                    step=None,
                    nnorm=1,
                    nvict=0,
                    method = "guess"
                )

                self.spec_dict[column_names[i]] = self.createSpectrumPropertyDict(column_names[i], energy, specData,
                                                                                  e0_init, pre1, pre2, post1, post2, 1)
        self.nomalizeSpectraAndPlot(self.spec_dict)
        self.spectrumDictToListWidget()

    def nomalizeSpectraAndPlot(self, spectrumParamDict):
        # print(self.spec_dict)
        try:
            self.spectrum_view.clear()
        except:
            pass
        self.spectrum_view.setLabel("bottom", "Energy")
        self.spectrum_view.setLabel("left", "Intensity")
        self.spectrum_view.addLegend()
        plt_colors = ['r', 'g', (31, 81, 255), 'c', 'm', 'y', 'w']
        for n, params in enumerate(spectrumParamDict.values()):
            e0_ = params['NormParams'][0]
            pre1_ = params['NormParams'][1]
            pre2_ = params['NormParams'][2]
            norm1_ = params['NormParams'][3]
            norm2_ = params['NormParams'][4]
            normOrder_ = params['NormParams'][5]

            preLine, postLine, self.normData = xanesNormalization(
                params['Data'][0],
                params['Data'][1],
                e0=e0_,
                step=None,
                nnorm=normOrder_,
                nvict=0,
                pre1=pre1_,
                pre2=pre2_,
                norm1=norm1_,
                norm2=norm2_
            )

            # 'NormParams': (e0, pre1, pre2, norm1, norm2, normOrder)}
            self.spectrum_view.plot(params['Data'][0], self.normData,
                                    pen=pg.mkPen(plt_colors[n], width=2),
                                    name=f"Norm._{params['Name']}")

    def loadAndPlotSpectrumData(self):
        filter = 'txt (*.tiff);;csv (*.csv)'
        file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Open a spectrum file", '',
                                                            'txt (*.tiff);;csv (*.csv);;all_files (*)', filter)

        if file_name[0]:
            self.spec_df = pd.read_csv(file_name[0], index_col=None)
            # print(self.spec_df.head())
            # print(self.spec_df.shape[1])
            self.createMultiSpectrumLibrary()
        else:
            return

    def spectrumDictToListWidget(self):
        for params in self.spec_dict.values():
            # Creates a QListWidgetItem
            specItem = QtWidgets.QListWidgetItem()

            # Setting QListWidgetItem Text
            specItem.setText(params['Name'])

            # Setting your QListWidgetItem Data
            specItem.setData(QtCore.Qt.UserRole, params)

            # Add the new rule to the QListWidget
            self.listWidget_Spectrum.addItem(specItem)

    def plotNormSpectrum(self, item):
        self.selectedItem = item
        self.editItemName = self.selectedItem.text()
        self.editItemData = self.selectedItem.data(QtCore.Qt.UserRole)

        e0_ = self.editItemData['NormParams'][0]
        pre1_ = self.editItemData['NormParams'][1]
        pre2_ = self.editItemData['NormParams'][2]
        norm1_ = self.editItemData['NormParams'][3]
        norm2_ = self.editItemData['NormParams'][4]
        normOrder_ = self.editItemData['NormParams'][5]

        self.dsb_norm_Eo.setValue(e0_)  # loop later
        self.dsb_norm_pre1.setValue(pre1_)
        self.dsb_norm_pre2.setValue(pre2_)
        self.dsb_norm_post1.setValue(norm1_)
        self.dsb_norm_post2.setValue(norm2_)
        self.sb_norm_order.setValue(normOrder_)

        preLine, postLine, normSpec = xanesNormalization(
            self.editItemData['Data'][0],
            self.editItemData['Data'][1],
            e0=e0_,
            step=None,
            nnorm=normOrder_,
            nvict=0,
            pre1=pre1_,
            pre2=pre2_,
            norm1=norm1_,
            norm2=norm2_
        )

        self.spectrum_view.clear()
        self.spectrum_view.plot(self.editItemData['Data'][0], self.editItemData['Data'][1],
                                title=f"Normalization Plot_{self.editItemData['Name']}",
                                pen=pg.mkPen('y', width=2), name=self.editItemData['Name'])
        self.spectrum_view.plot(self.editItemData['Data'][0], preLine, pen=pg.mkPen('c', width=2), name='Pre')
        self.spectrum_view.plot(self.editItemData['Data'][0], postLine, pen=pg.mkPen('m', width=2), name='Norm')

    # def updateNormParamaters(self):

    def updateSpecData(self, plotNorm=False):

        # loop later
        self.editItemData['NormParams'][0] = self.dsb_norm_Eo.value()
        self.editItemData['NormParams'][1] = self.dsb_norm_pre1.value()
        self.editItemData['NormParams'][2] = self.dsb_norm_pre2.value()
        self.editItemData['NormParams'][3] = self.dsb_norm_post1.value()
        self.editItemData['NormParams'][4] = self.dsb_norm_post2.value()
        self.editItemData['NormParams'][5] = self.sb_norm_order.value()

        self.spec_dict[self.editItemName] = self.editItemData
        self.selectedItem.setData(QtCore.Qt.UserRole, self.editItemData)
        if plotNorm:
            self.nomalizeSpectraAndPlot(self.spec_dict)
        else:
            self.plotNormSpectrum(self.selectedItem)

    def exportDisplayedSpectra(self):
        exporter = pg.exporters.CSVExporter(self.spectrum_view.plotItem)
        exporter.parameters()['columnMode'] = '(x,y,y,y) for all plots'
        file_name = QFileDialog().getSaveFileName(self, "save spectra", 'xanes.csv', 'spectra (*csv)')
        if file_name[0]:
            exporter.export(file_name[0])
        else:
            self.statusbar_main.showMessage('Saving cancelled')
            pass

class ChemMapViewer(MultiChannelWindow):

    def __init__(self, im_stack = None, 
                 e_list=None,
                 refs = None,
                 ref_names = None,
                 chem_map = None,
                 fit_params = None):
        
        super().__init__(image_dict=None)

        self.im_stack = im_stack
        self.e_list = e_list
        self.refs = refs
        self.ref_names = ref_names
        self.chem_map = chem_map
        self.fit_params = fit_params

        for n, (colorName, image) in enumerate(zip(cmap_dict.keys(), self.chem_map.transpose((2, 0, 1)))):
            low, high = np.min(image), np.max(image)
            self.image_dict[f'Image {n + 1}'] = {'ImageName': self.ref_names[n],
                                                       'ImageDir': '.',
                                                       'Image': image,
                                                       'Color': colorName,
                                                       'CmapLimits': (low, high),
                                                       'Opacity': 1.0
                                                       }

        uic.loadUi(os.path.join(ui_dir, 'MultiImageSpectrumView.ui'), self)
        self.user_wd = os.path.abspath("~")
        # Copy from MultiChannelWindow Start here
        self.canvas = self.img_view.addPlot(title="")
        self.canvas.getViewBox().invertY(True)
        #self.canvas.setZValue(-10)
        self.canvas.setAspectLocked(True)
        self.cb_choose_color.addItems([i for i in cmap_dict.keys()])
        #self.canvas.getViewBox().setBackgroundColor(pg.mkColor(222,222,222))
        #self.canvas.getViewBox().setOpacity(0.5)

        self.buildFromDictionary()

        self.actionLoad.triggered.connect(self.createMuliColorAndList)
        self.actionLoad_Stack.triggered.connect(self.createMuliColorAndList)
        self.cb_choose_color.currentTextChanged.connect(self.updateImageDictionary)
        self.pb_update_low_high.clicked.connect(self.updateImageDictionary)
        self.listWidget.itemClicked.connect(self.editImageProperties)
        self.listWidget.itemDoubleClicked.connect(self.showOneImageOnly)
        self.pb_show_selected.clicked.connect(self.showOneImageOnly)
        self.pb_show_all.clicked.connect(self.showAllItems)
        self.actionLoad_State_File.triggered.connect(self.importState)
        self.actionSave_State.triggered.connect(self.exportState)
        self.actionSave_View.triggered.connect(self.saveImage)


        # Copy from MultiChannelWindow End here
        self.actionSave_Spectrum_Data.triggered.connect(self.exportDisplayedSpectra)
        self.listWidget_Spectrum.itemClicked.connect(self.plotNormSpectrum)
        self.pb_apply_xanes_norm.clicked.connect(lambda: self.updateSpecData(plotNorm=True))
        self.createMultiSpectrumLibrary()

        [dsb.valueChanged.connect(lambda: self.updateSpecData()) for dsb in
         [self.dsb_norm_Eo, self.dsb_norm_pre1, self.dsb_norm_pre2, self.dsb_norm_post1,
          self.dsb_norm_post2, self.sb_norm_order]]

    def createSpectrumPropertyDict(self, specName, xdata, ydata, e0, pre1, pre2, norm1, norm2, normOrder):
        SingleSpecProperty = {'Name': specName,
                              'Data': (xdata, ydata),
                              'NormParams': [e0, pre1, pre2, norm1, norm2, normOrder]}

        return SingleSpecProperty

    def createMultiSpectrumLibrary(self):
        self.spec_dict = {}
        column_names = self.spec_df.columns
        spec_array = self.spec_df.to_numpy()
        energy = spec_array[:, 0]
        for i in range(self.spec_df.shape[1]):
            if i != 0:
                specData = spec_array[:, i]
                e0_init = energy[np.argmax(np.gradient(specData))]

                pre1, pre2, post1, post2 = xanesNormalization(
                    energy,
                    specData,
                    e0=e0_init,
                    step=None,
                    nnorm=1,
                    nvict=0,
                    method = "guess"
                )

                self.spec_dict[column_names[i]] = self.createSpectrumPropertyDict(column_names[i], energy, specData,
                                                                                  e0_init, pre1, pre2, post1, post2, 1)
        self.nomalizeSpectraAndPlot(self.spec_dict)
        self.spectrumDictToListWidget()

    def nomalizeSpectraAndPlot(self, spectrumParamDict):
        # print(self.spec_dict)
        try:
            self.spectrum_view.clear()
        except:
            pass
        self.spectrum_view.setLabel("bottom", "Energy")
        self.spectrum_view.setLabel("left", "Intensity")
        self.spectrum_view.addLegend()
        plt_colors = ['r', 'g', (31, 81, 255), 'c', 'm', 'y', 'w']
        for n, params in enumerate(spectrumParamDict.values()):
            e0_ = params['NormParams'][0]
            pre1_ = params['NormParams'][1]
            pre2_ = params['NormParams'][2]
            norm1_ = params['NormParams'][3]
            norm2_ = params['NormParams'][4]
            normOrder_ = params['NormParams'][5]

            preLine, postLine, self.normData = xanesNormalization(
                params['Data'][0],
                params['Data'][1],
                e0=e0_,
                step=None,
                nnorm=normOrder_,
                nvict=0,
                pre1=pre1_,
                pre2=pre2_,
                norm1=norm1_,
                norm2=norm2_
            )

            # 'NormParams': (e0, pre1, pre2, norm1, norm2, normOrder)}
            self.spectrum_view.plot(params['Data'][0], self.normData,
                                    pen=pg.mkPen(plt_colors[n], width=2),
                                    name=f"Norm._{params['Name']}")

    def loadAndPlotSpectrumData(self):
        filter = 'txt (*.tiff);;csv (*.csv)'
        file_name = QtWidgets.QFileDialog().getOpenFileName(self, "Open a spectrum file", '',
                                                            'txt (*.tiff);;csv (*.csv);;all_files (*)', filter)

        if file_name[0]:
            self.spec_df = pd.read_csv(file_name[0], index_col=None)
            # print(self.spec_df.head())
            # print(self.spec_df.shape[1])
            self.createMultiSpectrumLibrary()
        else:
            return

    def spectrumDictToListWidget(self):
        for params in self.spec_dict.values():
            # Creates a QListWidgetItem
            specItem = QtWidgets.QListWidgetItem()

            # Setting QListWidgetItem Text
            specItem.setText(params['Name'])

            # Setting your QListWidgetItem Data
            specItem.setData(QtCore.Qt.UserRole, params)

            # Add the new rule to the QListWidget
            self.listWidget_Spectrum.addItem(specItem)

    def plotNormSpectrum(self, item):
        self.selectedItem = item
        self.editItemName = self.selectedItem.text()
        self.editItemData = self.selectedItem.data(QtCore.Qt.UserRole)

        e0_ = self.editItemData['NormParams'][0]
        pre1_ = self.editItemData['NormParams'][1]
        pre2_ = self.editItemData['NormParams'][2]
        norm1_ = self.editItemData['NormParams'][3]
        norm2_ = self.editItemData['NormParams'][4]
        normOrder_ = self.editItemData['NormParams'][5]

        self.dsb_norm_Eo.setValue(e0_)  # loop later
        self.dsb_norm_pre1.setValue(pre1_)
        self.dsb_norm_pre2.setValue(pre2_)
        self.dsb_norm_post1.setValue(norm1_)
        self.dsb_norm_post2.setValue(norm2_)
        self.sb_norm_order.setValue(normOrder_)

        preLine, postLine, normSpec = xanesNormalization(
            self.editItemData['Data'][0],
            self.editItemData['Data'][1],
            e0=e0_,
            step=None,
            nnorm=normOrder_,
            nvict=0,
            pre1=pre1_,
            pre2=pre2_,
            norm1=norm1_,
            norm2=norm2_
        )

        self.spectrum_view.clear()
        self.spectrum_view.plot(self.editItemData['Data'][0], self.editItemData['Data'][1],
                                title=f"Normalization Plot_{self.editItemData['Name']}",
                                pen=pg.mkPen('y', width=2), name=self.editItemData['Name'])
        self.spectrum_view.plot(self.editItemData['Data'][0], preLine, pen=pg.mkPen('c', width=2), name='Pre')
        self.spectrum_view.plot(self.editItemData['Data'][0], postLine, pen=pg.mkPen('m', width=2), name='Norm')

    # def updateNormParamaters(self):

    def updateSpecData(self, plotNorm=False):

        # loop later
        self.editItemData['NormParams'][0] = self.dsb_norm_Eo.value()
        self.editItemData['NormParams'][1] = self.dsb_norm_pre1.value()
        self.editItemData['NormParams'][2] = self.dsb_norm_pre2.value()
        self.editItemData['NormParams'][3] = self.dsb_norm_post1.value()
        self.editItemData['NormParams'][4] = self.dsb_norm_post2.value()
        self.editItemData['NormParams'][5] = self.sb_norm_order.value()

        self.spec_dict[self.editItemName] = self.editItemData
        self.selectedItem.setData(QtCore.Qt.UserRole, self.editItemData)
        if plotNorm:
            self.nomalizeSpectraAndPlot(self.spec_dict)
        else:
            self.plotNormSpectrum(self.selectedItem)

    def exportDisplayedSpectra(self):
        exporter = pg.exporters.CSVExporter(self.spectrum_view.plotItem)
        exporter.parameters()['columnMode'] = '(x,y,y,y) for all plots'
        file_name = QFileDialog().getSaveFileName(self, "save spectra", 'xanes.csv', 'spectra (*csv)')
        if file_name[0]:
            exporter.export(file_name[0])
        else:
            self.statusbar_main.showMessage('Saving cancelled')
            pass

            
