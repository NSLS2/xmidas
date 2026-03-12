"""Dimensionality reduction, clustering, and classification functions.

All functions extracted from xmidas/utils/utils.py.
"""

import numpy as np
import scipy.stats as stats
import sklearn.decomposition as sd
import sklearn.cluster as sc


# ---------------------------------------------------------------------------
# Internal helpers (small re-uses from utils.py to avoid circular imports)
# ---------------------------------------------------------------------------

def _normalize(image_array, norm_point=-1):
    norm_stack = image_array / image_array[norm_point]
    im = np.array(norm_stack, dtype=np.float32)
    im[np.isnan(im)] = 0
    im[np.isinf(im)] = 0
    return im


def _remove_nan_inf(im):
    im = np.array(im, dtype=np.float32)
    im[np.isnan(im)] = 0
    im[np.isinf(im)] = 0
    return im


def _get_sum_spectra(image_array):
    return np.nansum(image_array, axis=(1, 2))


# ---------------------------------------------------------------------------
# Correlation-based classification
# ---------------------------------------------------------------------------

def classify(img_stack, correlation="Pearson"):
    """Compute a pixel-wise correlation map against the brightest pixel.

    Parameters
    ----------
    img_stack : 3D ndarray (nE, nX, nY)
    correlation : str
        "Pearson" or "Kendall".

    Returns
    -------
    cluster_image : 2D ndarray (nX, nY)
        Correlation map cubed.
    img_stack_ : 3D ndarray
        Original stack (unchanged).
    """
    img_stack_ = img_stack
    a, b, c = np.shape(img_stack_)
    norm_img_stack = _normalize(img_stack_)
    f = np.reshape(norm_img_stack, (a, (b * c)))

    max_x, max_y = np.where(norm_img_stack.sum(0) == (norm_img_stack.sum(0)).max())
    ref = norm_img_stack[:, int(max_x), int(max_y)]
    corr = np.zeros(len(f.T))
    for s in range(len(f.T)):
        if correlation == "Kendall":
            r, p = stats.kendalltau(ref, f.T[s])
        elif correlation == "Pearson":
            r, p = stats.pearsonr(ref, f.T[s])
        corr[s] = r

    cluster_image = np.reshape(corr, (b, c))
    return (cluster_image ** 3), img_stack_


def correlation_kmeans(img_stack, n_clusters, correlation="Pearson"):
    """Run KMeans on a Pearson/Kendall correlation image.

    Parameters
    ----------
    img_stack : 3D ndarray
    n_clusters : int
    correlation : str

    Returns
    -------
    X_cluster : 2D ndarray  (1-indexed cluster labels)
    """
    img, bg_image = classify(img_stack, correlation)
    img[np.isnan(img)] = -99999
    X = img.reshape((-1, 1))
    k_means = sc.KMeans(n_clusters)
    k_means.fit(X)
    X_cluster = k_means.labels_
    X_cluster = X_cluster.reshape(img.shape) + 1
    return X_cluster


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_stack(
    im_array, method="KMeans", n_clusters_=4, decomposed=False,
    decompose_method="PCA", decompose_comp=2
):
    """Cluster a 3D image stack spatially.

    Parameters
    ----------
    im_array : 3D ndarray (nE, nX, nY)
    method : str
        One of "KMeans", "MiniBatchKMeans", "MeanShift",
        "Spectral Clustering", "Affinity Propagation",
        "Correlation-Kmeans".
    n_clusters_ : int
    decomposed : bool
        If True, pre-denoise with decomposition before clustering.
    decompose_method : str
        Decomposition method passed to denoise_with_decomposition.
    decompose_comp : int
        Number of components for denoising.

    Returns
    -------
    decon_images : ndarray (n_clusters_, nX, nY)
    X_cluster : ndarray (nX, nY)
    decon_spectra : ndarray (nE, n_clusters_)
    """
    a, b, c = im_array.shape

    if method == "Correlation-Kmeans":
        X_cluster = correlation_kmeans(im_array, n_clusters_, correlation="Pearson")
    else:
        methods = {
            "MiniBatchKMeans": sc.MiniBatchKMeans,
            "KMeans": sc.KMeans,
            "MeanShift": sc.MeanShift,
            "Spectral Clustering": sc.SpectralClustering,
            "Affinity Propagation": sc.AffinityPropagation,
        }

        if decomposed:
            im_array = denoise_with_decomposition(
                im_array, method_=decompose_method, n_components=decompose_comp
            )

        flat_array = np.reshape(im_array, (a, (b * c)))
        init_cluster = methods[method](n_clusters=n_clusters_)
        init_cluster.fit(np.transpose(flat_array))
        X_cluster = init_cluster.labels_.reshape(b, c) + 1

    decon_spectra = np.zeros((a, n_clusters_))
    decon_images = np.zeros((n_clusters_, b, c))

    for i in range(n_clusters_):
        mask_i = np.where(X_cluster == (i + 1), X_cluster, 0)
        spec_i = _get_sum_spectra(im_array * mask_i)
        decon_spectra[:, i] = spec_i
        decon_images[i] = im_array.sum(0) * mask_i

    return decon_images, X_cluster, decon_spectra


def kmeans_variance(im_array):
    """Compute KMeans inertia for cluster counts 2–25 (elbow method).

    Returns
    -------
    clust_n : ndarray  cluster counts (2..25)
    var : ndarray      inertia for each count
    """
    a, b, c = im_array.shape
    flat_array = np.reshape(im_array, (a, (b * c)))
    var = np.arange(24)
    clust_n = np.arange(24) + 2

    for clust in var:
        init_cluster = sc.KMeans(n_clusters=int(clust + 2))
        init_cluster.fit(np.transpose(flat_array))
        var_ = init_cluster.inertia_
        var[clust] = np.float64(var_)

    return clust_n, var


# ---------------------------------------------------------------------------
# Dimensionality reduction / decomposition
# ---------------------------------------------------------------------------

def pca_scree(im_stack):
    """Compute TruncatedSVD singular values for a PCA scree plot.

    Parameters
    ----------
    im_stack : 3D ndarray (nE, nX, nY)

    Returns
    -------
    var : 1D ndarray  singular values
    """
    new_image = im_stack.transpose(2, 1, 0)
    x, y, z = np.shape(new_image)
    img_ = np.reshape(new_image, (x * y, z))
    pca = sd.TruncatedSVD(z - 1)
    pca.fit(img_)
    return pca.singular_values_


def decompose_stack(im_stack, decompose_method="PCA",
                    n_components_=3, generate_label_img=True):
    """Decompose a 3D stack using the selected method.

    Parameters
    ----------
    im_stack : 3D ndarray (nE, nX, nY)
    decompose_method : str
        One of PCA, IncrementalPCA, NMF, FastICA,
        DictionaryLearning, FactorAnalysis, TruncatedSVD.
    n_components_ : int
    generate_label_img : bool
        If True, compute deconvoluted spectra and a label map.

    Returns
    -------
    ims : ndarray (n_components_, nX, nY)
    spcs : ndarray
    decon_spetra : ndarray (nE, n_components_)
    decom_map : ndarray (nX, nY)
    """
    new_image = im_stack.transpose(2, 1, 0)
    new_image[new_image < 0] = 0

    x, y, z = np.shape(new_image)
    img_ = np.reshape(new_image, (x * y, z))
    methods_dict = {
        "PCA": sd.PCA,
        "IncrementalPCA": sd.IncrementalPCA,
        "NMF": sd.NMF,
        "FastICA": sd.FastICA,
        "DictionaryLearning": sd.MiniBatchDictionaryLearning,
        "FactorAnalysis": sd.FactorAnalysis,
        "TruncatedSVD": sd.TruncatedSVD,
    }

    _mdl = methods_dict[decompose_method](n_components=n_components_)
    ims = (_mdl.fit_transform(img_).reshape(x, y, n_components_)).transpose(2, 1, 0)
    spcs = _mdl.components_.transpose()
    decon_spetra = np.zeros((z, n_components_))
    decom_map = np.zeros((ims.shape))

    if generate_label_img:
        for i in range(n_components_):
            f = ims.copy()[i]
            f[f < 0] = 0
            f = np.where(f > 0 * np.std(f), f, 0)
            spec_i = ((new_image.T * f).sum(1)).sum(1)
            decon_spetra[:, i] = spec_i
            f[f > 0] = i + 1
            decom_map[i] = f
        decom_map = decom_map.sum(0)

    return np.float32(ims), spcs, decon_spetra, decom_map


def denoise_with_decomposition(img_stack, method_="PCA", n_components=4):
    """Denoise a stack by decomposing it and rebuilding a binary mask.

    Parameters
    ----------
    img_stack : 3D ndarray (nE, nX, nY)
    method_ : str
        Decomposition method (PCA, NMF, FastICA, etc.).
    n_components : int

    Returns
    -------
    filtered : 3D ndarray (float32)
    """
    new_image = img_stack.transpose(2, 1, 0)
    x, y, z = np.shape(new_image)
    img_ = np.reshape(new_image, (x * y, z))

    methods_dict = {
        "PCA": sd.PCA,
        "IncrementalPCA": sd.IncrementalPCA,
        "NMF": sd.NMF,
        "FastICA": sd.FastICA,
        "DictionaryLearning": sd.DictionaryLearning,
        "FactorAnalysis": sd.FactorAnalysis,
        "TruncatedSVD": sd.TruncatedSVD,
    }

    decomposed = methods_dict[method_](n_components=n_components)
    ims = (decomposed.fit_transform(img_).reshape(x, y, n_components)).transpose(2, 1, 0)
    ims[ims < 0] = 0
    ims[ims > 0] = 1
    mask = ims.sum(0)
    mask[mask > 1] = 1
    filtered = img_stack * mask
    return _remove_nan_inf(filtered)
