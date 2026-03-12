import logging
import sys
import os
import json
import scipy.stats as stats
import numpy as np
import pandas as pd
import tifffile as tf
import pyqtgraph as pg
import pyqtgraph.exporters
from glob import glob
from scipy.stats import linregress
from packaging import version

from PyQt6 import QtWidgets, QtCore, QtGui, uic, QtTest
from PyQt6.QtGui import QMovie
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QApplication
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool, PYQT_VERSION_STR

from xmidas.gui.windows.multichannel_viewer import MultiChannelWindow
from xmidas.utils import *
from xmidas.utils.color_maps import *
from xmidas.models.encoders import jsonEncoder
from xmidas.gui.windows.singleStackViewer import *
cmap_dict = create_color_maps()

ui_dir = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../layout"
))




class DecomposeViewer(QtWidgets.QMainWindow):
    mask_signal: pyqtSignal = QtCore.pyqtSignal(np.ndarray)
    mask_and_path_signal: pyqtSignal = QtCore.pyqtSignal(list)
    def __init__(self, to_decompose, im_stack, energy):
        super(DecomposeViewer, self).__init__()

        # Load the UI Page
        uic.loadUi(os.path.join(ui_dir, "decomposeViewer.ui"), self)
        self.centralwidget.setStyleSheet(open(os.path.join(ui_dir, "css/defaultStyle.css")).read())
        self.user_wd = os.path.abspath("~")
        
        self.to_decompose = to_decompose
        if im_stack is None:
            self.im_stack = self.to_decompose
        else:
            self.im_stack = im_stack
        self.energy = energy
        self.eigen_img = None
        self.eigen_spectra = None
        self.decon_spectra = None
        self.decon_map = None

        if self.to_decompose.ndim==2:
            self.to_decompose = np.expand_dims(self.to_decompose, axis=0)


        (self.dim1, self.dim3, self.dim2) = self.to_decompose.shape
        #self.hs_comp_number.setMaximum(self.dim1 - 1)
        self.eigen_image_view.ui.menuBtn.hide()
        self.eigen_image_view.ui.roiBtn.hide()
        self.eigen_image_view.setPredefinedGradient("viridis")

        self.eigen_masked_spectrum_view.setLabel("bottom", "Energy")
        self.eigen_masked_spectrum_view.setLabel("left", "Intensity", "A.U.")

        self.eigen_spectrum_view.setLabel("bottom", "Energy")
        self.eigen_spectrum_view.setLabel("left", "Weight", "A.U.")

        self.pb_view_stack.clicked.connect(self.view_stack)


        # connection
        self.pb_show_all.clicked.connect(lambda:self.show_all_spec(norm_to_max = True, add_offset = True))
        self.hs_comp_number.valueChanged.connect(self.update_plots)
        self.actionSave.triggered.connect(self.save_comp_data)
        self.pb_send_mask.clicked.connect(self.send_mask)
        #self.pb_openScatterPlot.clicked.connect(self.openScatterPlot)
        # self.pb_showMultiColor.clicked.connect(lambda: self.generateMultiColorView(withSpectra=False))
        # self.pb_showMultiImageXANESView.clicked.connect(lambda: self.generateMultiColorView(withSpectra=True))
        self.pb_calc_components.clicked.connect(self.decompose_and_display)
        self.pb_calc_cluster.clicked.connect(self.cluster_decomposed)
        self.sldr_mask_low.valueChanged.connect(lambda value: self.dsb_low_threshold.setValue(value / 100.0))
        self.sldr_mask_high.valueChanged.connect(lambda value: self.dsb_high_threshold.setValue(value / 100.0))
        self.pb_apply_threshold.clicked.connect(lambda:self.create_mask(self.hs_comp_number.value()))
        self.pb_browse_save_path.clicked.connect(self.browse_folder)
        self.pb_send_all_and_save.clicked.connect(self.send_mask_and_save)


    def view_stack(self):
        self.newWindow = singleStackViewer(self.im_stack)
        self.newWindow.show()

    def decompose_and_display(self):
            self.eigen_img,self.eigen_spectra,self.decon_spectra,self.decon_map = decompose_stack(
                self.to_decompose, 
                decompose_method=self.cb_comp_method.currentText(), 
                n_components_=self.sb_ncomp.value())
            
            self.hs_comp_number.setMaximum(self.eigen_img.shape[0]-1)
            self.update_plots(0)

    def cluster_decomposed(self):

        self.eigen_img, X_cluster, self.decon_spectra = cluster_stack(
                            self.to_decompose,
                            method=self.cb_clust_method.currentText(),
                            n_clusters_=self.sb_ncluster.value(),
                            decomposed=self.cb_use_comp_for_cluster.isChecked(),
                            decompose_method=self.cb_comp_method.currentText(),
                            decompose_comp=self.sb_ncomp.value())
        self.eigen_spectra = self.decon_spectra
        self.hs_comp_number.setMaximum((self.eigen_img.shape)[0]-1)
        self.update_plots(0)


    def update_plots(self, im_index):
        self.eigen_image_view.setImage(self.eigen_img[im_index])

        try:
            self.eigen_masked_spectrum_view.plot(self.energy, self.decon_spectra[:, im_index], clear=True)
            self.eigen_spectrum_view.plot(self.energy, self.eigen_spectra[:, im_index], clear=True)
        except:
            pass
        self.label_comp_number.setText(f"{im_index + 1}/{self.eigen_img.shape[0]}")
        #print(f"{self.eigen_img.shape = }")
        self.create_mask(im_index)

    def create_mask(self, im_index):

        self.threshold_low = self.dsb_low_threshold.value()
        self.threshold_high = self.dsb_high_threshold.value()
        self.dsb_low_threshold.setMaximum(self.threshold_high-1e-10)
        self.dsb_high_threshold.setMinimum(self.threshold_low+1e-10)

        self.norm_mask = remove_nan_inf(self.eigen_img[im_index]) / np.nanmax(self.eigen_img[im_index])
        self.norm_mask[(self.norm_mask < self.threshold_low) | 
                       (self.norm_mask > self.threshold_high)] = 0

        self.mask = self.norm_mask
        self.binary_mask = np.where(self.mask > 0, 1, 0)
        self.eigen_mask_image_view.setImage(self.binary_mask)

    def send_mask(self):

        """ Send masked xanes viewer"""

        if self.cb_use_binary_mask.isChecked():
            self.mask_signal.emit(self.im_stack*self.binary_mask[np.newaxis,:,:])
        else:
            self.mask_signal.emit(self.im_stack*self.mask[np.newaxis,:,:])
    
    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.le_save_path.setText(folder_path)

    def send_mask_and_save(self):

        save_path = os.path.join(self.le_save_path.text(), f"Cluster_{self.hs_comp_number.value()}")
        if self.cb_use_binary_mask.isChecked():
            self.mask_and_path_signal.emit([self.im_stack*self.binary_mask[np.newaxis,:,:], save_path])
        else:
            self.mask_and_path_signal.emit([self.im_stack*self.mask[np.newaxis,:,:], save_path])


    def show_all_spec(self, norm_to_max = True, add_offset = True):
        self.eigen_masked_spectrum_view.clear()
        self.plt_colors = ["g", "b", "r", "c", "m", "y", "w"] * 10
        offsets = np.arange(0, 2, 0.2)
        self.eigen_masked_spectrum_view.addLegend()
        for ii in range(self.decon_spectra.shape[1]):
            to_plot = self.decon_spectra[:, ii]
            if norm_to_max:
                to_plot = self.decon_spectra[:, ii] / self.decon_spectra[:, ii].max()
            if add_offset:
                to_plot = to_plot+ + offsets[ii]

            self.eigen_masked_spectrum_view.plot(
                self.energy,
                to_plot,
                pen=self.plt_colors[ii],
                name="component" + str(ii + 1),
            )
        self.eigen_spectrum_view.clear()
        self.eigen_spectrum_view.addLegend()
        for ii in range(self.eigen_spectra.shape[1]):
            to_plot = self.eigen_spectra[:, ii]
            if norm_to_max:
                to_plot = self.eigen_spectra[:, ii] / self.eigen_spectra[:, ii].max()
            if add_offset:
                to_plot = to_plot+ + offsets[ii]
            self.eigen_spectrum_view.plot(
                self.energy,
                to_plot,
                pen=self.plt_colors[ii],
                name="eigen_vector" + str(ii + 1),
            )

    def save_comp_data(self):
        file_name = QFileDialog().getSaveFileName(self, "save all data", self.user_wd, "data(*tiff *tif *txt *png )")
        if file_name[0]:
            self.show_all_spec(norm_to_max = False, add_offset = False)
            tf.imwrite(file_name[0] + "_eigen_weights.tiff", np.float32(self.self.eigen_img))
            tf.imwrite(file_name[0] + "_eigen_masks.tiff", np.float32(self.decon_map))
            exporter_spec = pg.exporters.CSVExporter(self.eigen_masked_spectrum_view.plotItem)
            exporter_spec.parameters()["columnMode"] = "(x,y) per plot"
            exporter_spec.export(file_name[0] + "_deconv_spec.csv")
            exporter_eigen = pg.exporters.CSVExporter(self.eigen_spectrum_view.plotItem)
            exporter_eigen.parameters()["columnMode"] = "(x,y) per plot"
            exporter_eigen.export(file_name[0] + "_eigen_vectors.csv")
            self.user_wd = os.path.dirname(file_name[0])
        else:
            pass

    def generateMultiColorView(self, withSpectra=False):
        self.multichanneldict = {}

        for n, (colorName, image) in enumerate(zip(cmap_dict.keys(), self.to_decompose.transpose(0, 1, 2))):
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

            # self.muli_color_window = MultiXANESWindow(image_dict=self.multichanneldict,
            #                                           spec_df=compXanesSpetraAll)
        else:
            self.muli_color_window = MultiChannelWindow(image_dict=self.multichanneldict)

        self.muli_color_window.show()

    # add energy column
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