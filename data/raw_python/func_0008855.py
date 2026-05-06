def _fit_island(self, island_data):
        """
        Take an Island, do all the parameter estimation and fitting.


        Parameters
        ----------
        island_data : :class:`AegeanTools.models.IslandFittingData`
            The island to be fit.

        Returns
        -------
        sources : list
            The sources that were fit.
        """
        global_data = self.global_data

        # global data
        dcurve = global_data.dcurve
        rmsimg = global_data.rmsimg

        # island data
        isle_num = island_data.isle_num
        idata = island_data.i
        innerclip, outerclip, max_summits = island_data.scalars
        xmin, xmax, ymin, ymax = island_data.offsets

        # get the beam parameters at the center of this island
        midra, middec = global_data.wcshelper.pix2sky([0.5 * (xmax + xmin), 0.5 * (ymax + ymin)])
        beam = global_data.psfhelper.get_psf_pix(midra, middec)
        del middec, midra

        icurve = dcurve[xmin:xmax, ymin:ymax]
        rms = rmsimg[xmin:xmax, ymin:ymax]

        is_flag = 0
        pixbeam = global_data.psfhelper.get_pixbeam_pixel((xmin + xmax) / 2., (ymin + ymax) / 2.)
        if pixbeam is None:
            # This island is not 'on' the sky, ignore it
            return []

        self.log.debug("=====")
        self.log.debug("Island ({0})".format(isle_num))

        params = self.estimate_lmfit_parinfo(idata, rms, icurve, beam, innerclip, outerclip, offsets=[xmin, ymin],
                                             max_summits=max_summits)

        # islands at the edge of a region of nans
        # result in no components
        if params is None or params['components'].value < 1:
            return []

        self.log.debug("Rms is {0}".format(np.shape(rms)))
        self.log.debug("Isle is {0}".format(np.shape(idata)))
        self.log.debug(" of which {0} are masked".format(sum(np.isnan(idata).ravel() * 1)))

        # Check that there is enough data to do the fit
        mx, my = np.where(np.isfinite(idata))
        non_blank_pix = len(mx)
        free_vars = len([1 for a in params.keys() if params[a].vary])
        if non_blank_pix < free_vars or free_vars == 0:
            self.log.debug("Island {0} doesn't have enough pixels to fit the given model".format(isle_num))
            self.log.debug("non_blank_pix {0}, free_vars {1}".format(non_blank_pix, free_vars))
            result = DummyLM()
            model = params
            is_flag |= flags.NOTFIT
        else:
            # Model is the fitted parameters
            fac = 1 / np.sqrt(2)
            if self.global_data.docov:
                C = Cmatrix(mx, my, pixbeam.a * FWHM2CC * fac, pixbeam.b * FWHM2CC * fac, pixbeam.pa)
                B = Bmatrix(C)
            else:
                C = B = None
            self.log.debug(
                "C({0},{1},{2},{3},{4})".format(len(mx), len(my), pixbeam.a * FWHM2CC, pixbeam.b * FWHM2CC, pixbeam.pa))
            errs = np.nanmax(rms)
            self.log.debug("Initial params")
            self.log.debug(params)
            result, _ = do_lmfit(idata, params, B=B)
            if not result.errorbars:
                is_flag |= flags.FITERR
            # get the real (sky) parameter errors
            model = covar_errors(result.params, idata, errs=errs, B=B, C=C)

            if self.global_data.dobias and self.global_data.docov:
                x, y = np.indices(idata.shape)
                acf = elliptical_gaussian(x, y, 1, 0, 0, pixbeam.a * FWHM2CC * fac, pixbeam.b * FWHM2CC * fac,
                                          pixbeam.pa)
                bias_correct(model, idata, acf=acf * errs ** 2)

            if not result.success:
                is_flag |= flags.FITERR

        self.log.debug("Final params")
        self.log.debug(model)

        # convert the fitting results to a list of sources [and islands]
        sources = self.result_to_components(result, model, island_data, is_flag)

        return sources