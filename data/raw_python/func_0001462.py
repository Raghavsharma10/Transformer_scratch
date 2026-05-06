def compute_bg(self, which_data="phase",
                   fit_offset="mean", fit_profile="tilt",
                   border_m=0, border_perc=0, border_px=0,
                   from_mask=None, ret_mask=False):
        """Compute background correction

        Parameters
        ----------
        which_data: str or list of str
            From which type of data to remove the background
            information. The list contains either "amplitude",
            "phase", or both.
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
        border_m: float
            Assume that a frame of `border_m` meters around the
            image is background. The value is converted to
            pixels and rounded.
        border_perc: float
            Assume that a frame of `border_perc` percent around
            the image is background. The value is converted to
            pixels and rounded. If the aspect ratio of the image
            is not one, then the average of the data's shape is
            used to compute the percentage in pixels.
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
        The `border_*` values are translated to pixel values and
        the largest pixel border is used to generate a mask
        image for background computation.

        If any of the `border_*` arguments are non-zero and
        `from_mask` is given, the intersection of the two
        is used, i.e. the positions where both, the frame
        mask and `from_mask`, are `True`.

        See Also
        --------
        qpimage.bg_estimate.estimate
        """
        which_data = QPImage._conv_which_data(which_data)
        # check validity
        if not ("amplitude" in which_data or
                "phase" in which_data):
            msg = "`which_data` must contain 'phase' or 'amplitude'!"
            raise ValueError(msg)
        # get border in px
        border_list = []
        if border_m:
            if border_m < 0:
                raise ValueError("`border_m` must be greater than zero!")
            border_list.append(border_m / self.meta["pixel size"])
        if border_perc:
            if border_perc < 0 or border_perc > 50:
                raise ValueError("`border_perc` must be in interval [0, 50]!")
            size = np.average(self.shape)
            border_list.append(size * border_perc / 100)
        if border_px:
            border_list.append(border_px)
        # get maximum border size
        if border_list:
            border_px = np.int(np.round(np.max(border_list)))
        elif from_mask is None:
            raise ValueError("Neither `from_mask` nor `border_*` given!")
        elif np.all(from_mask == 0):
            raise ValueError("`from_mask` must not be all-zero!")
        # Get affected image data
        imdat_list = []
        if "amplitude" in which_data:
            imdat_list.append(self._amp)
        if "phase" in which_data:
            imdat_list.append(self._pha)
        # Perform correction
        for imdat in imdat_list:
            mask = imdat.estimate_bg(fit_offset=fit_offset,
                                     fit_profile=fit_profile,
                                     border_px=border_px,
                                     from_mask=from_mask,
                                     ret_mask=ret_mask)
        return mask