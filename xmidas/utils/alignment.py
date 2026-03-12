"""Image alignment and registration functions using pystackreg."""

import numpy as np
from pystackreg import StackReg


def align_stack(
    stack_img, ref_image_void=True, ref_stack=None,
    transformation=StackReg.TRANSLATION, reference="previous"
):
    """Image registration flow using pystackreg.

    Parameters
    ----------
    stack_img : ndarray
        The image stack to align.
    ref_image_void : bool
        If True, use stack_img itself as the reference for registration.
        If False, use ref_stack as the reference.
    ref_stack : ndarray or None
        Reference stack to register against. Used only when ref_image_void=False.
    transformation : StackReg transformation constant
        Type of geometric transformation (e.g. TRANSLATION, RIGID_BODY, etc.).
    reference : str
        Reference frame strategy, e.g. "previous" or "first".

    Returns
    -------
    out_stk : ndarray (float32)
        Aligned image stack.
    tmats_ : ndarray
        Transformation matrices computed during registration.
    """
    sr = StackReg(transformation)

    if ref_image_void:
        tmats_ = sr.register_stack(stack_img, reference=reference)
    else:
        tmats_ = sr.register_stack(ref_stack, reference=reference)
        out_ref = sr.transform_stack(ref_stack)  # noqa: F841

    out_stk = sr.transform_stack(stack_img, tmats=tmats_)
    return np.float32(out_stk), tmats_


def align_simple(stack_img, transformation=StackReg.TRANSLATION, reference="previous"):
    """Simple iterative alignment using pystackreg.

    Registers the stack and applies the transformation 10 times.

    Parameters
    ----------
    stack_img : ndarray
        The image stack to align.
    transformation : StackReg transformation constant
        Type of geometric transformation.
    reference : str
        Reference frame strategy (currently unused in registration step).

    Returns
    -------
    out_stk : ndarray (float32)
        Aligned image stack.
    """
    sr = StackReg(transformation)
    tmats_ = sr.register_stack(stack_img, reference="previous")
    for i in range(10):
        out_stk = sr.transform_stack(stack_img, tmats=tmats_)
    return np.float32(out_stk)


def align_with_tmat(stack_img, tmat_file, transformation=StackReg.TRANSLATION):
    """Apply a pre-computed transformation matrix to align a stack.

    Parameters
    ----------
    stack_img : ndarray
        The image stack to transform.
    tmat_file : ndarray
        Pre-computed transformation matrices to apply.
    transformation : StackReg transformation constant
        Type of geometric transformation.

    Returns
    -------
    out_stk : ndarray (float32)
        Transformed image stack.
    """
    sr = StackReg(transformation)
    out_stk = sr.transform_stack(stack_img, tmats=tmat_file)
    return np.float32(out_stk)


def align_stack_iter(
    stack,
    ref_stack_void=True,
    ref_stack=None,
    transformation=StackReg.TRANSLATION,
    method=("previous", "first"),
    max_iter=2,
):
    """Iterative multi-pass alignment using pystackreg.

    Registers the stack repeatedly over multiple passes and reference methods
    to progressively improve alignment quality.

    Parameters
    ----------
    stack : ndarray
        The image stack to align.
    ref_stack_void : bool
        If True, use stack itself as the reference. If False, use ref_stack.
    ref_stack : ndarray or None
        External reference stack. Used only when ref_stack_void=False.
    transformation : StackReg transformation constant
        Type of geometric transformation.
    method : tuple of str
        Sequence of reference strategies to apply per iteration
        (e.g. ("previous", "first")).
    max_iter : int
        Number of outer alignment iterations.

    Returns
    -------
    stack : ndarray (float32)
        Aligned image stack.
    """
    if ref_stack_void:
        ref_stack = stack

    for i in range(max_iter):
        sr = StackReg(transformation)
        for ii in range(len(method)):
            print(ii, method[ii])
            tmats = sr.register_stack(ref_stack, reference=method[ii])
            ref_stack = sr.transform_stack(ref_stack)
            stack = sr.transform_stack(stack, tmats=tmats)

    return np.float32(stack)
