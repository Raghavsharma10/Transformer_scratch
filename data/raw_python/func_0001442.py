def estimate(data, fit_offset="mean", fit_profile="tilt",
             border_px=0, from_mask=None, ret_mask=False):
    """Estimate the background value of an image

    Parameters
    ----------
    data: np.ndarray
        Data from which to compute the background value
    fit_profile: str
        The type of background profile to fit:

        - "offset": offset only
        - "poly2o": 2D 2nd order polynomial with mixed terms
        - "tilt": 2D linear tilt with offset (default)
    fit_offset: str
        The method for computing the profile offset

        - "fit": offset as fitting parameter
        - "gauss": center of a gaussian fit
        - "mean": simple average
        - "mode": mode (see `qpimage.bg_estimate.mode`)
    border_px: float
        Assume that a frame of `border_px` pixels around
        the image is background.
    from_mask: boolean np.ndarray or None
        Use a boolean array to define the background area.
        The boolean mask must have the same shape as the
        input data. `True` elements are used for background
        estimation.
    ret_mask: bool
        Return the boolean mask used to compute the background.

    Notes
    -----
    If both `border_px` and `from_mask` are given, the
    intersection of the two is used, i.e. the positions
    where both, the frame mask and `from_mask`, are
    `True`.
    """
    if fit_profile not in VALID_FIT_PROFILES:
        msg = "`fit_profile` must be one of {}, got '{}'".format(
            VALID_FIT_PROFILES,
            fit_profile)
        raise ValueError(msg)
    if fit_offset not in VALID_FIT_OFFSETS:
        msg = "`fit_offset` must be one of {}, got '{}'".format(
            VALID_FIT_OFFSETS,
            fit_offset)
        raise ValueError(msg)
    # initial mask image
    if from_mask is not None:
        assert isinstance(from_mask, np.ndarray)
        mask = from_mask.copy()
    else:
        mask = np.ones_like(data, dtype=bool)
    # multiply with border mask image (intersection)
    if border_px > 0:
        border_px = int(np.round(border_px))
        mask_px = np.zeros_like(mask)
        mask_px[:border_px, :] = True
        mask_px[-border_px:, :] = True
        mask_px[:, :border_px] = True
        mask_px[:, -border_px:] = True
        # intersection
        np.logical_and(mask, mask_px, out=mask)
    # compute background image
    if fit_profile == "tilt":
        bgimg = profile_tilt(data, mask)
    elif fit_profile == "poly2o":
        bgimg = profile_poly2o(data, mask)
    else:
        bgimg = np.zeros_like(data, dtype=float)
    # add offsets
    if fit_offset == "fit":
        if fit_profile == "offset":
            msg = "`fit_offset=='fit'` only valid when `fit_profile!='offset`"
            raise ValueError(msg)
        # nothing else to do here, using offset from fit
    elif fit_offset == "gauss":
        bgimg += offset_gaussian((data - bgimg)[mask])
    elif fit_offset == "mean":
        bgimg += np.mean((data - bgimg)[mask])
    elif fit_offset == "mode":
        bgimg += offset_mode((data - bgimg)[mask])

    if ret_mask:
        ret = (bgimg, mask)
    else:
        ret = bgimg
    return ret