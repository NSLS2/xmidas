""" Helper Functions (make a class later)"""


import h5py
import logging
import numpy as np
import pandas as pd
import os

from scipy.signal import savgol_filter
from skimage.transform import resize

from xmidas.utils.alignment import (
    align_stack,
    align_simple,
    align_with_tmat,
    align_stack_iter,
)


logger = logging.getLogger()

def get_xrf_data(h='h5file'):
    """
    get xrf stack from h5 data generated at NSLS-II beamlines

     Arguments:
        h5/hdf5 file

     Returns:
         norm_xrf_stack -  xrf stack image normalized with Io
         mono_e  - excitation enegy used for xrf
         beamline - identity of the beamline
         Io_avg - an average Io value, used before taking log

    """

    f = h5py.File(h, "r")

    if list(f.keys())[0] == "xrfmap":
        logger.info("Data from HXN/TES/SRX")
        beamline = f["xrfmap/scan_metadata"].attrs["scan_instrument_id"]

        try:

            beamline_scalar = {"HXN": 2, "SRX": 0, "TES": 0}

            if beamline in beamline_scalar.keys():

                Io = np.array(f["xrfmap/scalers/val"])[:, :, beamline_scalar[beamline]]
                raw_xrf_stack = np.array(f["xrfmap/detsum/counts"])
                norm_xrf_stack = raw_xrf_stack
                Io_avg = int(remove_nan_inf(Io).mean())
            else:
                logger.error("Unknown Beamline Scalar")
        except Exception:
            logger.warning("Unknown Scalar: Raw Detector count in use")
            norm_xrf_stack = np.array(f["xrfmap/detsum/counts"])

    elif list(f.keys())[0] == "xrmmap":
        logger.info("Data from XFM")
        beamline = "XFM"
        raw_xrf_stack = np.array(f["xrmmap/mcasum/counts"])
        Io = np.array(f["xrmmap/scalars/I0"])
        norm_xrf_stack = raw_xrf_stack
        Io_avg = int(remove_nan_inf(Io).mean())

    elif list(f.keys())[0] == "MAPS":
        logger.info("MAPS file")
        beamline = "APS"
        raw_xrf_stack = np.array(f["MAPS/Spectra/mca_arr"])
        Io = 1 #have to find the name of the scalar
        norm_xrf_stack = raw_xrf_stack.transpose((1, 2, 0))
        Io_avg = int(remove_nan_inf(Io).mean())

    else:
        logger.error("Unknown Data Format")

    try:
        mono_e = int(f["xrfmap/scan_metadata"].attrs["instrument_mono_incident_energy"] * 1000)
        logger.info("Excitation energy was taken from the h5 data")

    except Exception:
        mono_e = 12000
        logger.info(f"Unable to get Excitation energy from the h5 data; using default value {mono_e} KeV")

    return remove_nan_inf(norm_xrf_stack.transpose((2, 0, 1))), mono_e + 1500, beamline, Io_avg


def remove_nan_inf(im):
    im = np.array(im, dtype=np.float32)
    im[np.isnan(im)] = 0
    im[np.isinf(im)] = 0
    return im


def rebin_image(im, bin_factor):
    arrx, arry = np.shape(im)
    if arrx / bin_factor != int or arrx / bin_factor != int:
        logger.error("Invalid Binning")

    else:
        shape = (arrx / bin_factor, arry / bin_factor)
        return im.reshape(shape).mean(-1).mean(1)


def remove_hot_pixels(image_array, NSigma=5):
    image_array = remove_nan_inf(image_array)
    a, b, c = np.shape(image_array)
    img_stack2 = np.zeros((a, b, c))
    for i in range(a):
        im = image_array[i, :, :]
        im[abs(im) > np.std(im) * NSigma] = im.mean()
        img_stack2[i, :, :] = im
    return img_stack2


def smoothen(image_array, w_size=5):
    a, b, c = np.shape(image_array)
    spec2D_Matrix = np.reshape(image_array, (a, (b * c)))
    smooth2D_Matrix = savgol_filter(spec2D_Matrix, w_size, w_size - 2, axis=0)
    return remove_nan_inf(np.reshape(smooth2D_Matrix, (a, b, c)))


def resize_stack(image_array, upscaling=False, scaling_factor=2):
    en, im1, im2 = np.shape(image_array)

    if upscaling:
        im1_ = im1 * scaling_factor
        im2_ = im2 * scaling_factor
        img_stack_resized = resize(image_array, (en, im1_, im2_))

    else:
        im1_ = int(im1 / scaling_factor)
        im2_ = int(im2 / scaling_factor)
        img_stack_resized = resize(image_array, (en, im1_, im2_))

    return img_stack_resized


def normalize(image_array, norm_point=-1):
    norm_stack = image_array / image_array[norm_point]
    return remove_nan_inf(norm_stack)


def remove_edges(image_array):
    # z, x, y = np.shape(image_array)
    return image_array[:, 1:-1, 1:-1]


def background_value(image_array):
    img = image_array.mean(0)
    img_h = img.mean(0)
    img_v = img.mean(1)
    h = np.gradient(img_h)
    v = np.gradient(img_v)
    bg = np.min([img_h[h == h.max()], img_v[v == v.max()]])
    return bg


def background_subtraction(img_stack, bg_percentage=10):
    img_stack = remove_nan_inf(img_stack)
    a, b, c = np.shape(img_stack)
    ref_image = np.reshape(img_stack.mean(0), (b * c))
    bg_ratio = int((b * c) * 0.01 * bg_percentage)
    bg_ = np.max(sorted(ref_image)[0:bg_ratio])
    bged_img_stack = img_stack - bg_[:, np.newaxis, np.newaxis]
    return bged_img_stack


def background_subtraction2(img_stack, bg_percentage=10):
    img_stack = remove_nan_inf(img_stack)
    a, b, c = np.shape(img_stack)
    bg_ratio = int((b * c) * 0.01 * bg_percentage)
    bged_img_stack = img_stack.copy()

    for n, img in enumerate(img_stack):
        bg_ = np.max(sorted(img.flatten())[0:bg_ratio])
        print(bg_)
        bged_img_stack[n] = img - bg_

    return remove_nan_inf(bged_img_stack)


def background1(img_stack):
    img = img_stack.sum(0)
    img_h = img.mean(0)
    img_v = img.mean(1)
    h = np.gradient(img_h)
    v = np.gradient(img_v)
    bg = np.min([img_h[h == h.max()], img_v[v == v.max()]])
    return bg


def get_sum_spectra(image_array):
    spec = np.nansum(image_array, axis=(1, 2))
    return spec


def get_mean_spectra(image_array):
    spec = np.nanmean(image_array, axis=(1, 2))
    return spec


def flatten_(image_array):
    z, x, y = np.shape(image_array)
    flat_array = np.reshape(image_array, (x * y, z))
    return flat_array


def image_to_pandas(image_array):
    a, b, c = np.shape(image_array)
    im_array = np.reshape(image_array, ((b * c), a))
    a, b = im_array.shape
    df = pd.DataFrame(
        data=im_array[:, :], columns=["e" + str(i) for i in range(b)], 
        index=["s" + str(i) for i in range(a)]
    )
    return df


def image_to_pandas2(image_array):
    a, b, c = np.shape(image_array)
    im_array = np.reshape(image_array, (a, (b * c)))
    a, b = im_array.shape
    df = pd.DataFrame(
        data=im_array[:, :], index=["e" + str(i) for i in range(a)], 
        columns=["s" + str(i) for i in range(b)]
    )
    return df


def neg_log(image_array):
    absorb = -1 * np.log(image_array)
    return remove_nan_inf(absorb)


def clean_stack(img_stack, auto_bg=False, bg_percentage=5):
    a, b, c = np.shape(img_stack)

    if auto_bg is True:
        bg_ = background1(img_stack)

    else:
        sum_spec = (img_stack.sum(1)).sum(1)
        ref_stk_num = np.where(sum_spec == sum_spec.max())[-1]

        ref_image = np.reshape(img_stack[ref_stk_num], (b * c))
        bg_ratio = int((b * c) * 0.01 * bg_percentage)
        bg_ = np.max(sorted(ref_image)[0:bg_ratio])

    bg = np.where(img_stack[ref_stk_num] > bg_, img_stack[ref_stk_num], 0)
    bg2 = np.where(bg < bg_, bg, 1)

    bged_img_stack = img_stack * bg2

    return remove_nan_inf(bged_img_stack)


def subtractBackground(im_stack, bg_region):
    if bg_region.ndim == 3:
        bg_region_ = np.mean(bg_region, axis=(1, 2))

    elif bg_region.ndim == 2:
        bg_region_ = np.mean(bg_region, axis=1)

    else:
        bg_region_ = bg_region

    return im_stack - bg_region_[:, np.newaxis, np.newaxis]




# XANES fitting, normalization, and reference-file functions have been moved to
# xmidas/utils/xanes_fitting.py
# (interploate_E, rfactor, rfactor_compute, fitting_admm, getStats,
#  xanes_fitting_1D, xanes_fitting, xanes_fitting_Line, xanes_fitting_Binned,
#  create_df_from_nor, create_df_from_nor_try2, energy_from_logfile,
#  xanesNormalization, xanesNormStack, getDeconvolutedXANESSpectrum)



# Alignment functions have been moved to xmidas/utils/alignment.py.
# They are re-exported here for backward compatibility.
# align_stack, align_simple, align_with_tmat, align_stack_iter

def normalize_and_scale(stack):
    return (stack**2/stack.sum(0)[np.newaxis,:,:])/stack.max()


def applyMaskGetMeanSpectrum(im_stack, mask):
    """A 2d mask to multiply with the 3d xanes stack and returns mean spectrum"""

    masked_stack = im_stack * mask
    return get_mean_spectra(masked_stack)


