def priorized_fit_islands(self, filename, catalogue, hdu_index=0, outfile=None, bkgin=None, rmsin=None, cores=1,
                              rms=None, bkg=None, beam=None, lat=None, imgpsf=None, catpsf=None, stage=3, ratio=None, outerclip=3,
                              doregroup=True, docov=True, cube_index=None):
        """
        Take an input catalog, and image, and optional background/noise images
        fit the flux and ra/dec for each of the given sources, keeping the morphology fixed

        if doregroup is true the groups will be recreated based on a matching radius/probability.
        if doregroup is false then the islands of the input catalog will be preserved.

        Multiple cores can be specified, and will be used.


        Parameters
        ----------
        filename : str or HDUList
            Image filename or HDUList.

        catalogue : str or list
            Input catalogue file name or list of OutputSource objects.

        hdu_index : int
            The index of the FITS HDU (extension).

        outfile : str
            file for printing catalog (NOT a table, just a text file of my own design)

        rmsin, bkgin : str or HDUList
            Filename or HDUList for the noise and background images.
            If either are None, then it will be calculated internally.

        cores : int
            Number of CPU cores to use. None means all cores.

        rms : float
            Use this rms for the entire image (will also assume that background is 0)

        beam : (major, minor, pa)
            Floats representing the synthesised beam (degrees).
            Replaces whatever is given in the FITS header.
            If the FITS header has no BMAJ/BMIN then this is required.

        lat : float
            The latitude of the telescope (declination of zenith).

        imgpsf : str or HDUList
             Filename or HDUList for a psf image.

        catpsf : str or HDUList
             Filename or HDUList for the catalogue psf image.

        stage : int
            Refitting stage

        ratio : float
            If not None - ratio of image psf to catalog psf, otherwise interpret from catalogue or image if possible

        innerclip, outerclip : float
            The seed (inner) and flood (outer) clipping level (sigmas).

        docov : bool
            If True then include covariance matrix in the fitting process. (default=True)

        cube_index : int
            For image cubes, slice determines which slice is used.


        Returns
        -------
        sources : list
            List of sources measured.

        """

        from AegeanTools.cluster import regroup

        self.load_globals(filename, hdu_index=hdu_index, bkgin=bkgin, rmsin=rmsin, rms=rms, bkg=bkg, cores=cores, verb=True,
                          do_curve=False, beam=beam, lat=lat, psf=imgpsf, docov=docov, cube_index=cube_index)

        global_data = self.global_data
        far = 10 * global_data.beam.a  # degrees
        # load the table and convert to an input source list
        if isinstance(catalogue, six.string_types):
            input_table = load_table(catalogue)
            input_sources = np.array(table_to_source_list(input_table))
        else:
            input_sources = np.array(catalogue)

        if len(input_sources) < 1:
            self.log.debug("No input sources for priorized fitting")
            return []

        # reject sources with missing params
        ok = True
        for param in ['ra', 'dec', 'peak_flux', 'a', 'b', 'pa']:
            if np.isnan(getattr(input_sources[0], param)):
                self.log.info("Source 0, is missing param '{0}'".format(param))
                ok = False
        if not ok:
            self.log.error("Missing parameters! Not fitting.")
            self.log.error("Maybe your table is missing or mis-labeled columns?")
            return []
        del ok

        src_mask = np.ones(len(input_sources), dtype=bool)

        # check to see if the input catalog contains psf information
        has_psf = getattr(input_sources[0], 'psf_a', None) is not None

        # the input sources are the initial conditions for our fits.
        # Expand each source size if needed.

        # If ratio is provided we just the psf by this amount
        if ratio is not None:
            self.log.info("Using ratio of {0} to scale input source shapes".format(ratio))
            far *= ratio
            for i, src in enumerate(input_sources):
                # Sources with an unknown psf are rejected as they are either outside the image
                # or outside the region covered by the psf
                skybeam = global_data.psfhelper.get_beam(src.ra, src.dec)
                if skybeam is None:
                    src_mask[i] = False
                    self.log.info("Excluding source ({0.island},{0.source}) due to lack of psf knowledge".format(src))
                    continue
                # the new source size is the previous size, convolved with the expanded psf
                src.a = np.sqrt(src.a ** 2 + (skybeam.a * 3600) ** 2 * (1 - 1 / ratio ** 2))
                src.b = np.sqrt(src.b ** 2 + (skybeam.b * 3600) ** 2 * (1 - 1 / ratio ** 2))
                # source with funky a/b are also rejected
                if not np.all(np.isfinite((src.a, src.b))):
                    self.log.info("Excluding source ({0.island},{0.source}) due to funky psf ({0.a},{0.b},{0.pa})".format(src))
                    src_mask[i] = False

        # if we know the psf from the input catalogue (has_psf), or if it was provided via a psf map
        # then we use that psf.
        elif catpsf is not None or has_psf:
            if catpsf is not None:
                self.log.info("Using catalog PSF from {0}".format(catpsf))
                psf_helper = PSFHelper(catpsf, None)  # might need to set the WCSHelper to be not None
            else:
                self.log.info("Using catalog PSF from input catalog")
                psf_helper = None
            for i, src in enumerate(input_sources):
                if (src.psf_a <=0) or (src.psf_b <=0):
                    src_mask[i] = False
                    self.log.info("Excluding source ({0.island},{0.source}) due to psf_a/b <=0".format(src))
                    continue
                if has_psf:
                    catbeam = Beam(src.psf_a / 3600, src.psf_b / 3600, src.psf_pa)
                else:
                    catbeam = psf_helper.get_beam(src.ra, src.dec)
                imbeam = global_data.psfhelper.get_beam(src.ra, src.dec)
                # If either of the above are None then we skip this source.
                if catbeam is None or imbeam is None:
                    src_mask[i] = False
                    self.log.info("Excluding source ({0.island},{0.source}) due to lack of psf knowledge".format(src))
                    continue

                # TODO: The following assumes that the various psf's are scaled versions of each other
                # and makes no account for differing position angles. This needs to be checked and/or addressed.

                # deconvolve the source shape from the catalogue psf
                src.a = (src.a / 3600) ** 2 - catbeam.a ** 2 + imbeam.a ** 2  # degrees

                # clip the minimum source shape to be the image psf
                if src.a < 0:
                    src.a = imbeam.a * 3600  # arcsec
                else:
                    src.a = np.sqrt(src.a) * 3600  # arcsec

                src.b = (src.b / 3600) ** 2 - catbeam.b ** 2 + imbeam.b ** 2
                if src.b < 0:
                    src.b = imbeam.b * 3600  # arcsec
                else:
                    src.b = np.sqrt(src.b) * 3600  # arcsec
        else:
            self.log.info("Not scaling input source sizes")

        self.log.info("{0} sources in catalog".format(len(input_sources)))
        self.log.info("{0} sources accepted".format(sum(src_mask)))

        if len(src_mask) < 1:
            self.log.debug("No sources accepted for priorized fitting")
            return []

        input_sources = input_sources[src_mask]
        # redo the grouping if required
        if doregroup:
            groups = regroup(input_sources, eps=np.sqrt(2), far=far)
        else:
            groups = list(island_itergen(input_sources))

        if cores == 1:  # single-threaded, no parallel processing
            queue = []
        else:
            queue = pprocess.Queue(limit=cores, reuse=1)
            fit_parallel = queue.manage(pprocess.MakeReusable(self._refit_islands))

        sources = []
        island_group = []
        group_size = 20

        for i, island in enumerate(groups):
            island_group.append(island)
            # If the island group is full queue it for the subprocesses to fit
            if len(island_group) >= group_size:
                if cores > 1:
                    fit_parallel(island_group, stage, outerclip, istart=i)
                else:
                    res = self._refit_islands(island_group, stage, outerclip, istart=i)
                    queue.append(res)
                island_group = []

        # The last partially-filled island group also needs to be queued for fitting
        if len(island_group) > 0:
            if cores > 1:
                fit_parallel(island_group, stage, outerclip, istart=i)
            else:
                res = self._refit_islands(island_group, stage, outerclip, istart=i)
                queue.append(res)

        # now unpack the fitting results in to a list of sources
        for s in queue:
            sources.extend(s)

        sources = sorted(sources)

        # Write the output to the output file
        if outfile:
            print(header.format("{0}-({1})".format(__version__, __date__), filename), file=outfile)
            print(OutputSource.header, file=outfile)

        components = 0
        for source in sources:
            if isinstance(source, OutputSource):
                components += 1
                if outfile:
                    print(str(source), file=outfile)

        self.log.info("fit {0} components".format(components))
        self.sources.extend(sources)
        return sources