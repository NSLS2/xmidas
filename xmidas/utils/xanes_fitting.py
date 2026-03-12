"""XANES fitting, normalization, and reference-file utilities.

All functions here have been extracted from xmidas/utils/utils.py.
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
from skimage import filters
from sklearn import linear_model

from xmidas.utils.larch_norm import pre_edge_simple


# ---------------------------------------------------------------------------
# Internal helpers (small re-uses from utils.py to avoid circular imports)
# ---------------------------------------------------------------------------

def _remove_nan_inf(im):
    im = np.array(im, dtype=np.float32)
    im[np.isnan(im)] = 0
    im[np.isinf(im)] = 0
    return im


def _resize_stack(image_array, upscaling=False, scaling_factor=2):
    from skimage.transform import resize
    en, im1, im2 = np.shape(image_array)
    if upscaling:
        im1_, im2_ = im1 * scaling_factor, im2 * scaling_factor
    else:
        im1_, im2_ = int(im1 / scaling_factor), int(im2 / scaling_factor)
    return resize(image_array, (en, im1_, im2_))


def _get_mean_spectra(image_array):
    return np.nanmean(image_array, axis=(1, 2))


# ---------------------------------------------------------------------------
# Energy interpolation
# ---------------------------------------------------------------------------

def interploate_E(refs, e):
    """Interpolate reference spectra onto the measured energy grid."""
    n = np.shape(refs)[1]
    refs = np.array(refs)
    ref_e = refs[:, 0]
    ref = refs[:, 1:n]
    all_ref = []
    for i in range(n - 1):
        ref_i = np.interp(e, ref_e, ref[:, i])
        all_ref.append(ref_i)
    return np.array(all_ref)


# ---------------------------------------------------------------------------
# R-factor metrics
# ---------------------------------------------------------------------------

def rfactor(spectrum_experimental, spectrum_fit):
    r"""Compute R-factor between two spectra.

    Parameters
    ----------
    spectrum_experimental : ndarray
        Observed spectrum (N elements).
    spectrum_fit : ndarray
        Fitted spectrum (N elements).

    Returns
    -------
    float
        R-factor value.
    """
    dif = spectrum_experimental - spectrum_fit
    dif_sum = np.sum(np.abs(dif), axis=0)
    data_sum = np.sum(np.abs(spectrum_experimental), axis=0)
    data_sum = np.clip(data_sum, a_min=1e-30, a_max=None)
    return dif_sum / data_sum


def rfactor_compute(spectrum, fit_results, ref_spectra):
    r"""Compute R-factor from fitting results and reference spectra.

    Parameters
    ----------
    spectrum : ndarray
        Observed spectrum, 1D or 2D.
    fit_results : ndarray
        Fitting coefficients, same ndim as spectrum.
    ref_spectra : 2D ndarray
        Reference spectra array (NxK).

    Returns
    -------
    float
        R-factor value.
    """
    assert (
        spectrum.ndim == 1 or spectrum.ndim == 2
    ), "Parameter 'spectrum' must be 1D or 2D array, ({spectrum.ndim})"
    assert spectrum.ndim == fit_results.ndim, (
        f"Spectrum data (ndim = {spectrum.ndim}) and fitting results "
        f"(ndim = {fit_results.ndim}) must have the same number of dimensions"
    )
    assert ref_spectra.ndim == 2, "Parameter 'ref_spectra' must be 2D array, ({ref_spectra.ndim})"
    assert spectrum.shape[0] == ref_spectra.shape[0], (
        f"Arrays 'spectrum' ({spectrum.shape}) and 'ref_spectra' ({ref_spectra.shape}) "
        "must have the same number of data points"
    )
    assert fit_results.shape[0] == ref_spectra.shape[1], (
        f"Arrays 'fit_results' ({fit_results.shape}) and 'ref_spectra' ({ref_spectra.shape}) "
        "must have the same number of spectrum points"
    )
    if spectrum.ndim == 2:
        assert spectrum.shape[1] == fit_results.shape[1], (
            f"Arrays 'spectrum' {spectrum.shape} and 'fit_results' {fit_results.shape}"
            "must have the same number of columns"
        )
    spectrum_fit = np.matmul(ref_spectra, fit_results)
    return rfactor(spectrum, spectrum_fit)


# ---------------------------------------------------------------------------
# ADMM fitting
# ---------------------------------------------------------------------------

def fitting_admm(data, ref_spectra, *, rate=0.2, maxiter=100, epsilon=1e-30,
                 non_negative=True, weight_to_whiteline=False):
    r"""Fit multiple spectra using the ADMM algorithm.

    Parameters
    ----------
    data : ndarray(float), 2D
        Observed spectra, shape (K, N).
    ref_spectra : ndarray(float), 2D
        Reference spectra, shape (K, Q).
    rate : float
        Descent rate (1/lambda).
    maxiter : int
        Maximum number of iterations.
    epsilon : float
        Convergence criterion threshold.
    non_negative : bool
        If True, solution is constrained to be non-negative.
    weight_to_whiteline : bool
        Apply custom weighting to emphasise the white-line region.

    Returns
    -------
    w : ndarray (Q, N)
        Fitting coefficients.
    rfactor : ndarray
        R-factor map.
    convergence : ndarray
        Convergence history.
    feasibility : ndarray
        Feasibility history.

    Notes
    -----
    Prototype originally implemented by Hanfei Yan in Matlab.
    """
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)
    assert ref_spectra.ndim == 2, "Data array 'ref_spectra' must have 2 dimensions"

    n_pts = data.shape[0]
    n_pixels = data.shape[1]
    n_pts_2 = ref_spectra.shape[0]
    n_refs = ref_spectra.shape[1]

    assert (
        n_pts == n_pts_2
    ), f"ADMM fitting: number of spectrum points in data ({n_pts}) and references ({n_pts_2}) do not match."
    assert rate > 0.0, f"ADMM fitting: parameter 'rate' is zero or negative ({rate:.6g})"
    assert maxiter > 0, f"ADMM fitting: parameter 'maxiter' is zero or negative ({rate})"
    assert epsilon > 0.0, f"ADMM fitting: parameter 'epsilon' is zero or negative ({rate:.6g})"

    wgt = np.ones(len(data))

    if weight_to_whiteline:
        wgt[0:-15] = 0.4
        wgt[-25:-1] = 0.5

    y = data
    A = ref_spectra
    At = np.transpose(A)  # noqa: F841

    z = A.T @ np.diag(wgt) @ y
    c = A.T @ np.diag(wgt) @ A

    w = np.ones(shape=[n_refs, n_pixels])
    u = np.zeros(shape=[n_refs, n_pixels])

    convergence = np.zeros(shape=[maxiter])
    feasibility = np.zeros(shape=[maxiter])

    dg = np.eye(n_refs, dtype=float) * rate
    m1 = np.linalg.inv((c + dg))

    n_iter = 0
    for i in range(maxiter):
        m2 = z + (w - u) * rate
        x = np.matmul(m1, m2)
        w_updated = x + u
        if non_negative:
            w_updated = w_updated.clip(min=0)
        u = u + x - w_updated

        conv = np.linalg.norm(w_updated - w) / np.linalg.norm(w_updated)
        convergence[i] = conv
        feasibility[i] = np.linalg.norm(x - w_updated)

        w = w_updated

        if conv < epsilon:
            n_iter = i + 1
            break

    r = rfactor_compute(data, w, ref_spectra)

    convergence = convergence[:n_iter]
    feasibility = feasibility[:n_iter]

    return w, r, convergence, feasibility


# ---------------------------------------------------------------------------
# Fitting statistics
# ---------------------------------------------------------------------------

def getStats(spec, fit, num_refs=2):
    """Compute goodness-of-fit statistics.

    Returns a dict with R_Factor, R_Square, Chi_Square, Reduced Chi_Square.
    """
    stats = {}
    SS_tot = np.sum((spec - np.mean(spec)) ** 2)
    SS_res = np.sum((spec - fit) ** 2)
    r_factor = 1 - (SS_res / SS_tot)
    stats["R_Factor"] = np.around(r_factor, 5)

    r_square = 1 - (SS_res / SS_tot)
    stats["R_Square"] = np.around(r_square, 4)

    chisq = np.sum(((spec - fit) ** 2) / np.var(spec))
    stats["Chi_Square"] = np.around(chisq, 5)

    red_chisq = chisq / (len(spec) - num_refs)
    stats["Reduced Chi_Square"] = np.around(red_chisq, 5)

    return stats


# ---------------------------------------------------------------------------
# Core fitting functions
# ---------------------------------------------------------------------------

def xanes_fitting_1D(spec, e_list, refs, method="NNLS", alphaForLM=0.01):
    """Linear combination fit of a single spectrum against reference standards.

    Parameters
    ----------
    spec : 1D ndarray
        Measured spectrum.
    e_list : 1D ndarray
        Energy points of spec.
    refs : 2D ndarray
        Reference spectra array (energy + standards columns).
    method : str
        One of "NNLS", "LASSO", "RIDGE", "ADMM".
    alphaForLM : float
        Regularisation parameter for LASSO/RIDGE/ADMM.

    Returns
    -------
    stats : dict
        Goodness-of-fit statistics.
    coeffs : ndarray
        Fitted coefficients for each reference.
    """
    spec = np.nan_to_num(spec)
    refs = np.nan_to_num(refs)
    int_refs = interploate_E(refs, e_list)

    if method == "NNLS":
        coeffs, r = opt.nnls(int_refs.T, spec)

    elif method == "LASSO":
        lasso = linear_model.Lasso(positive=True, alpha=alphaForLM)
        fit_results = lasso.fit(int_refs.T, spec)
        coeffs = fit_results.coef_

    elif method == "RIDGE":
        ridge = linear_model.Ridge(alpha=alphaForLM)
        fit_results = ridge.fit(int_refs.T, spec)
        coeffs = fit_results.coef_

    elif method == "ADMM":
        coeffs, r_factor, convergence, feasibility = fitting_admm(
            spec, int_refs.T, maxiter=100, rate=alphaForLM, epsilon=1e-30
        )
        coeffs = np.squeeze(coeffs.T)

    fit = np.dot(coeffs, int_refs)
    stats = getStats(spec, fit, num_refs=np.min(np.shape(int_refs.T)))

    return stats, coeffs


def xanes_fitting(im_stack, e_list, refs, method="NNLS", alphaForLM=0.1, binStack=False):
    """Linear combination fit of a 3D image stack against reference standards.

    Parameters
    ----------
    im_stack : 3D ndarray, shape (nE, nX, nY)
        XANES image stack.
    e_list : 1D ndarray
        Energy axis.
    refs : 2D ndarray
        Reference spectra.
    method : str
        Fitting method (NNLS, LASSO, RIDGE, ADMM).
    alphaForLM : float
        Regularisation parameter.
    binStack : bool
        If True, bin stack by factor 4 before fitting.

    Returns
    -------
    abundance_map : ndarray (nX, nY, nRefs)
    r_factor_im : ndarray (nX, nY)
    mean_coeffs : ndarray (nRefs,)
    """
    if binStack:
        im_stack = _resize_stack(im_stack, scaling_factor=4)

    en, im1, im2 = np.shape(im_stack)
    im_array = im_stack.reshape(en, im1 * im2)
    coeffs_arr = []
    r_factor_arr = []
    lasso = linear_model.Lasso(positive=True, alpha=alphaForLM)  # noqa: F841

    if not method == "ADMM":
        for n, i in enumerate(range(im1 * im2)):
            stats, coeffs = xanes_fitting_1D(
                im_array[:, i], e_list, refs, method=method, alphaForLM=alphaForLM
            )
            coeffs_arr.append(coeffs)
            r_factor_arr.append(stats["R_Factor"])

        abundance_map = np.reshape(coeffs_arr, (im1, im2, -1))
        r_factor_im = np.reshape(r_factor_arr, (im1, im2))

    elif method == "ADMM":
        int_refs = interploate_E(refs, e_list)
        coeffs_arr, r_factor_im, convergence, feasibility = fitting_admm(
            im_array, int_refs.T, maxiter=100, rate=alphaForLM, epsilon=1e-30
        )
        abundance_map = np.reshape((coeffs_arr.T), (im1, im2, -1))

    return abundance_map, r_factor_im, np.mean(coeffs_arr, axis=0)


def xanes_fitting_Line(im_stack, e_list, refs, method="NNLS", alphaForLM=0.05):
    """Linear combination fit averaged along the second spatial axis (line fit).

    Returns
    -------
    meanStats : dict
    mean_coeffs : ndarray
    """
    en, im1, im2 = np.shape(im_stack)
    im_array = np.mean(im_stack, 2)
    coeffs_arr = []
    meanStats = {"R_Factor": 0, "R_Square": 0, "Chi_Square": 0, "Reduced Chi_Square": 0}

    for i in range(im1):
        stats, coeffs = xanes_fitting_1D(
            im_array[:, i], e_list, refs, method=method, alphaForLM=alphaForLM
        )
        coeffs_arr.append(coeffs)
        for key in stats.keys():
            meanStats[key] += stats[key]

    for key, vals in meanStats.items():
        meanStats[key] = np.around((vals / im1), 5)

    return meanStats, np.mean(coeffs_arr, axis=0)


def xanes_fitting_Binned(im_stack, e_list, refs, method="NNLS", alphaForLM=0.05):
    """Linear combination fit on a binned stack, skipping low-intensity pixels.

    Returns
    -------
    meanStats : dict
    mean_coeffs : ndarray
    """
    im_stack = _resize_stack(im_stack, scaling_factor=10)
    val = filters.threshold_otsu(im_stack[-1])
    en, im1, im2 = np.shape(im_stack)
    im_array = im_stack.reshape(en, im1 * im2)
    coeffs_arr = []
    meanStats = {"R_Factor": 0, "R_Square": 0, "Chi_Square": 0, "Reduced Chi_Square": 0}

    specs_fitted = 0
    total_spec = im1 * im2
    for i in range(total_spec):
        spec = im_array[:, i]
        if spec[-1] > val:
            specs_fitted += 1
            stats, coeffs = xanes_fitting_1D(
                spec / spec[-1], e_list, refs, method=method, alphaForLM=alphaForLM
            )
            coeffs_arr.append(coeffs)
            for key in stats.keys():
                meanStats[key] += stats[key]

    for key, vals in meanStats.items():
        meanStats[key] = np.around((vals / specs_fitted), 6)
    return meanStats, np.mean(coeffs_arr, axis=0)


# ---------------------------------------------------------------------------
# Reference file readers
# ---------------------------------------------------------------------------

def create_df_from_nor(athenafile="fe_refs.nor"):
    """Create a pandas DataFrame from an Athena .nor file.

    First column is energy; column headers are sample names.
    """
    refs = np.loadtxt(athenafile)
    n_refs = refs.shape[-1]
    skip_raw_n = n_refs + 6

    df = pd.read_table(
        athenafile, sep=r'\s+', skiprows=skip_raw_n, header=None,
        usecols=np.arange(0, n_refs)
    )
    df2 = pd.read_table(
        athenafile, sep=r'\s+', skiprows=skip_raw_n - 1,
        usecols=np.arange(0, n_refs + 1)
    )
    new_col = df2.columns.drop("#")
    df.columns = new_col
    return df, list(new_col)


def create_df_from_nor_try2(athenafile="fe_refs.nor"):
    """Create a pandas DataFrame from an Athena .nor file (alternate parser)."""
    refs = np.loadtxt(athenafile)
    n_refs = refs.shape[-1]
    df_refs = pd.DataFrame(refs)

    df = pd.read_csv(athenafile, header=None)
    new_col = list((str(df.iloc[n_refs + 5].values)).split(" ")[2::2])
    df_refs.columns = new_col

    return df_refs, list(new_col)


def energy_from_logfile(logfile="maps_log_tiff.txt"):
    """Extract energy values from a MAPS log file."""
    df = pd.read_csv(logfile, header=None, sep=r'\s+', skiprows=9)
    return df[9][df[7] == "energy"].values.astype(float)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def xanesNormalization(e, mu, e0=7125, nnorm=2, nvict=0,
                       pre1=None, pre2=-50, norm1=None, norm2=None,
                       useFlattened=False, Elemline="Fe_K"):
    """Normalize a single XANES spectrum.

    Parameters
    ----------
    e, mu       : energy & absorption arrays
    e0          : edge energy (float)
    nnorm       : degree of post-edge polynomial
    nvict       : exponent for pre-edge fit weighting
    pre1, pre2  : relative pre-edge fitting bounds
    norm1, norm2: relative post-edge fitting bounds
    useFlattened: ignored in this version
    Elemline    : placeholder, no longer used

    Returns
    -------
    pre_edge_arr, post_edge_arr, norm_arr : ndarrays
    """
    res = pre_edge_simple(
        e, mu,
        e0=e0, nnorm=nnorm, nvict=nvict,
        pre1=pre1, pre2=pre2,
        norm1=norm1, norm2=norm2
    )
    return res["pre_edge"], res["post_edge"], res["norm"]


def xanesNormStack(e_list, im_stack,
                   e0=7125, nnorm=2, nvict=0,
                   pre1=None, pre2=-50,
                   norm1=None, norm2=None,
                   ignorePostEdgeNorm=False):
    """Apply XANES normalization to every spectrum in a 3D stack.

    Parameters
    ----------
    e_list : 1D array
        Energy grid.
    im_stack : 3D array, shape (nE, nX, nY)
        Raw absorption stack.
    e0, nnorm, nvict, pre1, pre2, norm1, norm2 : fit params
        Passed to pre_edge_simple.
    ignorePostEdgeNorm : bool
        If True, multiply normalized spectrum by post-edge baseline.

    Returns
    -------
    normed_stack : 3D array, same shape as im_stack
    """
    nE, nX, nY = im_stack.shape
    flat_in = im_stack.reshape(nE, -1)
    flat_out = np.zeros_like(flat_in)

    for i in range(flat_in.shape[1]):
        spec = flat_in[:, i]
        res = pre_edge_simple(
            e_list, spec,
            e0=e0, nnorm=nnorm, nvict=nvict,
            pre1=pre1, pre2=pre2,
            norm1=norm1, norm2=norm2
        )
        norm_spec = res["norm"]
        post_edge = res["post_edge"]

        if ignorePostEdgeNorm:
            flat_out[:, i] = norm_spec * post_edge
        else:
            flat_out[:, i] = norm_spec

    return _remove_nan_inf(flat_out.reshape(nE, nX, nY))


# ---------------------------------------------------------------------------
# Spectral decomposition analysis
# ---------------------------------------------------------------------------

def getDeconvolutedXANESSpectrum(xanesStack, chemMapStack, energy, clusterSigma=1):
    """Extract component XANES spectra weighted by cluster/chemical maps.

    Parameters
    ----------
    xanesStack : 3D ndarray (nE, nX, nY)
    chemMapStack : 3D ndarray (nComponents, nX, nY)
    energy : 1D ndarray
    clusterSigma : float
        Threshold in units of std-dev for masking.

    Returns
    -------
    compXanesSpetraAll : DataFrame
    """
    compXanesSpetraAll = pd.DataFrame()
    compXanesSpetraAll['Energy'] = energy

    for n, compImage in enumerate(chemMapStack):
        mask = np.where(compImage > clusterSigma * np.std(compImage), compImage, 0)
        compXanesSpetraAll[f'Component_{n + 1}'] = _get_mean_spectra(xanesStack * mask)

    return compXanesSpetraAll
