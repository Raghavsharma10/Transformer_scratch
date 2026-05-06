def find_sources_in_image(self, filename, hdu_index=0, outfile=None, rms=None, bkg=None, max_summits=None, innerclip=5,
                              outerclip=4, cores=None, rmsin=None, bkgin=None, beam=None, doislandflux=False,
                              nopositive=False, nonegative=False, mask=None, lat=None, imgpsf=None, blank=False,
                              docov=True, cube_index=None):
        """
        Run the Aegean source finder.


        Parameters
        ----------
        filename : str or HDUList
            Image filename or HDUList.

        hdu_index : int
            The index of the FITS HDU (extension).

        outfile : str
            file for printing catalog (NOT a table, just a text file of my own design)

        rms : float
            Use this rms for the entire image (will also assume that background is 0)

        max_summits : int
            Fit up to this many components to each island (extras are included but not fit)

        innerclip, outerclip : float
            The seed (inner) and flood (outer) clipping level (sigmas).

        cores : int
            Number of CPU cores to use. None means all cores.

        rmsin, bkgin : str or HDUList
            Filename or HDUList for the noise and background images.
            If either are None, then it will be calculated internally.

        beam : (major, minor, pa)
            Floats representing the synthesised beam (degrees).
            Replaces whatever is given in the FITS header.
            If the FITS header has no BMAJ/BMIN then this is required.

        doislandflux : bool
            If True then each island will also be characterized.

        nopositive, nonegative : bool
            Whether to return positive or negative sources.
            Default nopositive=False, nonegative=True.

        mask : str
            The filename of a region file created by MIMAS.
            Islands outside of this region will be ignored.

        lat : float
            The latitude of the telescope (declination of zenith).

        imgpsf : str or HDUList
             Filename or HDUList for a psf image.

        blank : bool
            Cause the output image to be blanked where islands are found.

        docov : bool
            If True then include covariance matrix in the fitting process. (default=True)

        cube_index : int
            For image cubes, cube_index determines which slice is used.

        Returns
        -------
        sources : list
            List of sources found.
        """

        # Tell numpy to be quiet
        np.seterr(invalid='ignore')
        if cores is not None:
            if not (cores >= 1): raise AssertionError("cores must be one or more")

        self.load_globals(filename, hdu_index=hdu_index, bkgin=bkgin, rmsin=rmsin, beam=beam, rms=rms, bkg=bkg, cores=cores,
                          verb=True, mask=mask, lat=lat, psf=imgpsf, blank=blank, docov=docov, cube_index=cube_index)
        global_data = self.global_data
        rmsimg = global_data.rmsimg
        data = global_data.data_pix

        self.log.info("beam = {0:5.2f}'' x {1:5.2f}'' at {2:5.2f}deg".format(
            global_data.beam.a * 3600, global_data.beam.b * 3600, global_data.beam.pa))
        # stop people from doing silly things.
        if outerclip > innerclip:
            outerclip = innerclip
        self.log.info("seedclip={0}".format(innerclip))
        self.log.info("floodclip={0}".format(outerclip))

        isle_num = 0

        if cores == 1:  # single-threaded, no parallel processing
            queue = []
        else:
            queue = pprocess.Queue(limit=cores, reuse=1)
            fit_parallel = queue.manage(pprocess.MakeReusable(self._fit_islands))

        island_group = []
        group_size = 20
        for i, xmin, xmax, ymin, ymax in self._gen_flood_wrap(data, rmsimg, innerclip, outerclip, domask=True):
            # ignore empty islands
            # This should now be impossible to trigger
            if np.size(i) < 1:
                self.log.warn("Empty island detected, this should be imposisble.")
                continue
            isle_num += 1
            scalars = (innerclip, outerclip, max_summits)
            offsets = (xmin, xmax, ymin, ymax)
            island_data = IslandFittingData(isle_num, i, scalars, offsets, doislandflux)
            # If cores==1 run fitting in main process. Otherwise build up groups of islands
            # and submit to queue for subprocesses. Passing a group of islands is more
            # efficient than passing single islands to the subprocesses.
            if cores == 1:
                res = self._fit_island(island_data)
                queue.append(res)
            else:
                island_group.append(island_data)
                # If the island group is full queue it for the subprocesses to fit
                if len(island_group) >= group_size:
                    fit_parallel(island_group)
                    island_group = []

        # The last partially-filled island group also needs to be queued for fitting
        if len(island_group) > 0:
            fit_parallel(island_group)

        # Write the output to the output file
        if outfile:
            print(header.format("{0}-({1})".format(__version__, __date__), filename), file=outfile)
            print(OutputSource.header, file=outfile)

        sources = []
        for srcs in queue:
            if srcs:  # ignore empty lists
                for src in srcs:
                    # ignore sources that we have been told to ignore
                    if (src.peak_flux > 0 and nopositive) or (src.peak_flux < 0 and nonegative):
                        continue
                    sources.append(src)
                    if outfile:
                        print(str(src), file=outfile)
        self.sources.extend(sources)
        return sources