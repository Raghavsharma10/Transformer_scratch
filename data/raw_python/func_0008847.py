def estimate_lmfit_parinfo(self, data, rmsimg, curve, beam, innerclip, outerclip=None, offsets=(0, 0),
                               max_summits=None):
        """
        Estimates the number of sources in an island and returns initial parameters for the fit as well as
        limits on those parameters.

        Parameters
        ----------
        data : 2d-array
            (sub) image of flux values. Background should be subtracted.

        rmsimg : 2d-array
            Image of 1sigma values

        curve : 2d-array
            Image of curvature values [-1,0,+1]

        beam : :class:`AegeanTools.fits_image.Beam`
            The beam information for the image.

        innerclip, outerclip : float
            Inerr and outer level for clipping (sigmas).

        offsets : (int, int)
            The (x,y) offset of data within it's parent image

        max_summits : int
            If not None, only this many summits/components will be fit. More components may be
            present in the island, but subsequent components will not have free parameters.

        Returns
        -------
        model : lmfit.Parameters
            The initial estimate of parameters for the components within this island.
        """

        debug_on = self.log.isEnabledFor(logging.DEBUG)
        is_flag = 0
        global_data = self.global_data

        # check to see if this island is a negative peak since we need to treat such cases slightly differently
        isnegative = max(data[np.where(np.isfinite(data))]) < 0
        if isnegative:
            self.log.debug("[is a negative island]")

        if outerclip is None:
            outerclip = innerclip

        self.log.debug(" - shape {0}".format(data.shape))

        if not data.shape == curve.shape:
            self.log.error("data and curvature are mismatched")
            self.log.error("data:{0} curve:{1}".format(data.shape, curve.shape))
            raise AssertionError()

        # For small islands we can't do a 6 param fit
        # Don't count the NaN values as part of the island
        non_nan_pix = len(data[np.where(np.isfinite(data))].ravel())
        if 4 <= non_nan_pix <= 6:
            self.log.debug("FIXED2PSF")
            is_flag |= flags.FIXED2PSF
        elif non_nan_pix < 4:
            self.log.debug("FITERRSMALL!")
            is_flag |= flags.FITERRSMALL
        else:
            is_flag = 0
        if debug_on:
            self.log.debug(" - size {0}".format(len(data.ravel())))

        if min(data.shape) <= 2 or (is_flag & flags.FITERRSMALL) or (is_flag & flags.FIXED2PSF):
            # 1d islands or small islands only get one source
            if debug_on:
                self.log.debug("Tiny summit detected")
                self.log.debug("{0}".format(data))
            summits = [[data, 0, data.shape[0], 0, data.shape[1]]]
            # and are constrained to be point sources
            is_flag |= flags.FIXED2PSF
        else:
            if isnegative:
                # the summit should be able to include all pixels within the island not just those above innerclip
                kappa_sigma = np.where(curve > 0.5, np.where(data + outerclip * rmsimg < 0, data, np.nan), np.nan)
            else:
                kappa_sigma = np.where(-1 * curve > 0.5, np.where(data - outerclip * rmsimg > 0, data, np.nan), np.nan)
            summits = list(self._gen_flood_wrap(kappa_sigma, np.ones(kappa_sigma.shape), 0, domask=False))

        params = lmfit.Parameters()
        i = 0
        summits_considered = 0
        # This can happen when the image contains regions of nans
        # the data/noise indicate an island, but the curvature doesn't back it up.
        if len(summits) < 1:
            self.log.debug("Island has {0} summits".format(len(summits)))
            return None

        # add summits in reverse order of peak SNR - ie brightest first
        for summit, xmin, xmax, ymin, ymax in sorted(summits, key=lambda x: np.nanmax(-1. * abs(x[0]))):
            summits_considered += 1
            summit_flag = is_flag
            if debug_on:
                self.log.debug(
                    "Summit({5}) - shape:{0} x:[{1}-{2}] y:[{3}-{4}]".format(summit.shape, ymin, ymax, xmin, xmax, i))
            try:
                if isnegative:
                    amp = np.nanmin(summit)
                    xpeak, ypeak = np.unravel_index(np.nanargmin(summit), summit.shape)
                else:
                    amp = np.nanmax(summit)
                    xpeak, ypeak = np.unravel_index(np.nanargmax(summit), summit.shape)
            except ValueError as e:
                if "All-NaN" in e.message:
                    self.log.warn("Summit of nan's detected - this shouldn't happen")
                    continue
                else:
                    raise e

            if debug_on:
                self.log.debug(" - max is {0:f}".format(amp))
                self.log.debug(" - peak at {0},{1}".format(xpeak, ypeak))
            yo = ypeak + ymin
            xo = xpeak + xmin

            # Summits are allowed to include pixels that are between the outer and inner clip
            # This means that sometimes we get a summit that has all it's pixels below the inner clip
            # So we test for that here.
            snr = np.nanmax(abs(data[xmin:xmax + 1, ymin:ymax + 1] / rmsimg[xmin:xmax + 1, ymin:ymax + 1]))
            if snr < innerclip:
                self.log.debug("Summit has SNR {0} < innerclip {1}: skipping".format(snr, innerclip))
                continue

            # allow amp to be 5% or (innerclip) sigma higher
            # TODO: the 5% should depend on the beam sampling
            # note: when innerclip is 400 this becomes rather stupid
            if amp > 0:
                amp_min, amp_max = 0.95 * min(outerclip * rmsimg[xo, yo], amp), amp * 1.05 + innerclip * rmsimg[xo, yo]
            else:
                amp_max, amp_min = 0.95 * max(-outerclip * rmsimg[xo, yo], amp), amp * 1.05 - innerclip * rmsimg[xo, yo]

            if debug_on:
                self.log.debug("a_min {0}, a_max {1}".format(amp_min, amp_max))

            pixbeam = global_data.psfhelper.get_pixbeam_pixel(yo + offsets[0], xo + offsets[1])
            if pixbeam is None:
                self.log.debug(" Summit has invalid WCS/Beam - Skipping.")
                continue

            # set a square limit based on the size of the pixbeam
            xo_lim = 0.5 * np.hypot(pixbeam.a, pixbeam.b)
            yo_lim = xo_lim

            yo_min, yo_max = yo - yo_lim, yo + yo_lim
            # if yo_min == yo_max:  # if we have a 1d summit then allow the position to vary by +/-0.5pix
            #    yo_min, yo_max = yo_min - 0.5, yo_max + 0.5

            xo_min, xo_max = xo - xo_lim, xo + xo_lim
            # if xo_min == xo_max:  # if we have a 1d summit then allow the position to vary by +/-0.5pix
            #    xo_min, xo_max = xo_min - 0.5, xo_max + 0.5

            # the size of the island
            xsize = data.shape[0]
            ysize = data.shape[1]

            # initial shape is the psf
            sx = pixbeam.a * FWHM2CC
            sy = pixbeam.b * FWHM2CC

            # lmfit does silly things if we start with these two parameters being equal
            sx = max(sx, sy * 1.01)

            # constraints are based on the shape of the island
            # sx,sy can become flipped so we set the min/max account for this
            sx_min, sx_max = sy * 0.8, max((max(xsize, ysize) + 1) * math.sqrt(2) * FWHM2CC, sx * 1.1)
            sy_min, sy_max = sy * 0.8, max((max(xsize, ysize) + 1) * math.sqrt(2) * FWHM2CC, sx * 1.1)

            theta = pixbeam.pa  # Degrees
            flag = summit_flag

            # check to see if we are going to fit this component
            if max_summits is not None:
                maxxed = i >= max_summits
            else:
                maxxed = False

            # components that are not fit need appropriate flags
            if maxxed:
                summit_flag |= flags.NOTFIT
                summit_flag |= flags.FIXED2PSF

            if debug_on:
                self.log.debug(" - var val min max | min max")
                self.log.debug(" - amp {0} {1} {2} ".format(amp, amp_min, amp_max))
                self.log.debug(" - xo {0} {1} {2} ".format(xo, xo_min, xo_max))
                self.log.debug(" - yo {0} {1} {2} ".format(yo, yo_min, yo_max))
                self.log.debug(" - sx {0} {1} {2} | {3} {4}".format(sx, sx_min, sx_max, sx_min * CC2FHWM,
                                                                    sx_max * CC2FHWM))
                self.log.debug(" - sy {0} {1} {2} | {3} {4}".format(sy, sy_min, sy_max, sy_min * CC2FHWM,
                                                                    sy_max * CC2FHWM))
                self.log.debug(" - theta {0} {1} {2}".format(theta, -180, 180))
                self.log.debug(" - flags {0}".format(flag))
                self.log.debug(" - fit?  {0}".format(not maxxed))

            # TODO: figure out how incorporate the circular constraint on sx/sy
            prefix = "c{0}_".format(i)
            params.add(prefix + 'amp', value=amp, min=amp_min, max=amp_max, vary=not maxxed)
            params.add(prefix + 'xo', value=xo, min=float(xo_min), max=float(xo_max), vary=not maxxed)
            params.add(prefix + 'yo', value=yo, min=float(yo_min), max=float(yo_max), vary=not maxxed)

            if summit_flag & flags.FIXED2PSF > 0:
                psf_vary = False
            else:
                psf_vary = not maxxed
            params.add(prefix + 'sx', value=sx, min=sx_min, max=sx_max, vary=psf_vary)
            params.add(prefix + 'sy', value=sy, min=sy_min, max=sy_max, vary=psf_vary)
            params.add(prefix + 'theta', value=theta, vary=psf_vary)
            params.add(prefix + 'flags', value=summit_flag, vary=False)

            # starting at zero allows the maj/min axes to be fit better.
            # if params[prefix + 'theta'].vary:
            #     params[prefix + 'theta'].value = 0

            i += 1
        if debug_on:
            self.log.debug("Estimated sources: {0}".format(i))
        # remember how many components are fit.
        params.add('components', value=i, vary=False)
        # params.components=i
        if params['components'].value < 1:
            self.log.debug("Considered {0} summits, accepted {1}".format(summits_considered, i))
        return params