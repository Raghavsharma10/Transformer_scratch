def estimate_bg(self, fit_offset="mean", fit_profile="tilt",
                    border_px=0, from_mask=None, ret_mask=False):
        """Estimate image background

        Parameters
        ----------
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
            The mask image must have the same shape as the
            input data.`True` elements are used for background
            estimation.
        ret_mask: bool
            Return the mask image used to compute the background.

        Notes
        -----
        If both `border_px` and `from_mask` are given, the
        intersection of the two resulting mask images is used.

        The arguments passed to this method are stored in the
        hdf5 file `self.h5` and are used for optional integrity
        checking using `qpimage.integrity_check.check`.

        See Also
        --------
        qpimage.bg_estimate.estimate
        """
        # remove existing bg before accessing imdat.image
        self.set_bg(bg=None, key="fit")
        # compute bg
        bgimage, mask = bg_estimate.estimate(data=self.image,
                                             fit_offset=fit_offset,
                                             fit_profile=fit_profile,
                                             border_px=border_px,
                                             from_mask=from_mask,
                                             ret_mask=True)
        attrs = {"fit_offset": fit_offset,
                 "fit_profile": fit_profile,
                 "border_px": border_px}
        self.set_bg(bg=bgimage, key="fit", attrs=attrs)
        # save `from_mask` separately (arrays vs. h5 attributes)
        # (if `from_mask` is `None`, this will remove the array)
        self["estimate_bg_from_mask"] = from_mask
        # return mask image
        if ret_mask:
            return mask