import numpy as np

def preedge(energy, mu, e0=None, npre=1, pre1=None, pre2=None, nvict=0):
    energy = np.asarray(energy)
    mu = np.asarray(mu)

    # 1) determine e0
    if e0 is None:
        deriv = np.gradient(mu, energy)
        ie0 = np.nanargmax(np.abs(deriv))
        e0 = energy[ie0]
    else:
        ie0 = np.argmin(np.abs(energy - e0))
        e0 = energy[ie0]

    # 2) default regions
    dE = energy[1] - energy[0]
    pre1 = -2 * dE if pre1 is None else pre1
    pre2 = -0.5 * dE if pre2 is None else pre2

    # 3) ensure ordering
    if pre1 > pre2:
        pre1, pre2 = pre2, pre1

    low, high = e0 + pre1, e0 + pre2
    mask = (energy >= low) & (energy <= high)

    # 4) fallback if too few points
    if np.sum(mask) < (npre + 1):
        idx0 = max(ie0 - (npre + 1), 0)
        mask = np.zeros_like(mask, bool)
        mask[idx0:ie0] = True

    x = energy[mask]
    y = mu[mask] * x**nvict

    # 5) fit baseline
    if npre == 0 or x.size == 0:
        val = np.mean(y) if y.size else np.mean(mu[:2])
        pre_edge = np.full_like(mu, val)
    else:
        coef = np.polyfit(x, y, 1)
        baseline = np.polyval(coef, energy)
        pre_edge = baseline * energy**(-nvict)

    return e0, ie0, pre_edge


def normalize(energy, mu, pre_edge, e0,
              norm1=None, norm2=None, nnorm=1):
    energy = np.asarray(energy)
    mu = np.asarray(mu)
    dE = energy[1] - energy[0]

    # 1) default norms
    norm1 = 2 * dE if norm1 is None else norm1
    norm2 = 10 * dE if norm2 is None else norm2

    # 2) ensure ordering
    if norm1 > norm2:
        norm1, norm2 = norm2, norm1

    low, high = e0 + norm1, e0 + norm2
    mask = (energy >= low) & (energy <= high)

    # 3) fallback to points just above edge
    min_pts = nnorm + 1
    if np.sum(mask) < min_pts:
        i0 = np.argmin(np.abs(energy - e0))
        start = min(i0 + 1, len(energy) - min_pts)
        mask = np.zeros_like(mask, bool)
        mask[start:start + min_pts] = True

    x = energy[mask]
    y = (mu - pre_edge)[mask]

    # 4) fit poly
    coef = ([0] * (nnorm + 1) if x.size == 0
            else np.polyfit(x, y, nnorm))
    poly = np.polyval(coef, energy)
    post_edge = pre_edge + poly

    # 5) compute step & norm
    ie0 = np.argmin(np.abs(energy - e0))
    edge_step = post_edge[ie0] - pre_edge[ie0]
    edge_step = abs(edge_step) if edge_step != 0 else 1e-12
    norm = (mu - pre_edge) / edge_step

    return post_edge, norm, edge_step



def pre_edge_simple(energy, mu, **kwargs):
    """
    Combined pre-edge subtraction and normalization.

    Returns
    -------
    dict with keys: e0, edge_step, pre_edge, post_edge, norm
    """
    e0, ie0, pre_edge_arr = preedge(energy, mu,
                                    e0=kwargs.get('e0', None),
                                    npre=kwargs.get('npre', 1),
                                    pre1=kwargs.get('pre1', None),
                                    pre2=kwargs.get('pre2', None),
                                    nvict=kwargs.get('nvict', 0))
    post_edge, norm_arr, edge_step = normalize(
        energy, mu, pre_edge_arr, e0,
        norm1=kwargs.get('norm1', None),
        norm2=kwargs.get('norm2', None),
        nnorm=kwargs.get('nnorm', 1)
    )
    return {
        'e0': e0,
        'edge_step': edge_step,
        'pre_edge': pre_edge_arr,
        'post_edge': post_edge,
        'norm': norm_arr
    }
