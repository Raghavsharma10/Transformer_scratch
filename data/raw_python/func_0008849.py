def load_globals(self, filename, hdu_index=0, bkgin=None, rmsin=None, beam=None, verb=False, rms=None, bkg=None, cores=1,
                     do_curve=True, mask=None, lat=None, psf=None, blank=False, docov=True, cube_index=None):
        """
        Populate the global_data object by loading or calculating the various components

        Parameters
        ----------
        filename : str or HDUList
            Main image which source finding is run on

        hdu_index : int
            HDU index of the image within the fits file, default is 0 (first)

        bkgin, rmsin : str or HDUList
            background and noise image filename or HDUList

        beam : :class:`AegeanTools.fits_image.Beam`
            Beam object representing the synthsized beam. Will replace what is in the FITS header.

        verb : bool
            Verbose. Write extra lines to INFO level log.

        rms, bkg : float
            A float that represents a constant rms/bkg levels for the entire image.
            Default = None, which causes the rms/bkg to be loaded or calculated.

        cores : int
            Number of cores to use if different from what is autodetected.


        do_curve : bool
            If True a curvature map will be created, default=True.

        mask : str or :class:`AegeanTools.regions.Region`
            filename or Region object

        lat : float
            Latitude of the observing telescope (declination of zenith)

        psf : str or HDUList
            Filename or HDUList of a psf image

        blank : bool
            True = blank output image where islands are found.
            Default = False.

        docov : bool
            True = use covariance matrix in fitting.
            Default = True.

        cube_index : int
            For an image cube, which slice to use.

        """
        # don't reload already loaded data
        if self.global_data.img is not None:
            return
        img = FitsImage(filename, hdu_index=hdu_index, beam=beam, cube_index=cube_index)
        beam = img.beam

        debug = logging.getLogger('Aegean').isEnabledFor(logging.DEBUG)

        if mask is None:
            self.global_data.region = None
        else:
            # allow users to supply and object instead of a filename
            if isinstance(mask, Region):
                self.global_data.region = mask
            elif os.path.exists(mask):
                self.log.info("Loading mask from {0}".format(mask))
                self.global_data.region = Region.load(mask)
            else:
                self.log.error("File {0} not found for loading".format(mask))
                self.global_data.region = None

        self.global_data.wcshelper = WCSHelper.from_header(img.get_hdu_header(), beam, lat)
        self.global_data.psfhelper = PSFHelper(psf, self.global_data.wcshelper)

        self.global_data.beam = self.global_data.wcshelper.beam
        self.global_data.img = img
        self.global_data.data_pix = img.get_pixels()
        self.global_data.dtype = type(self.global_data.data_pix[0][0])
        self.global_data.bkgimg = np.zeros(self.global_data.data_pix.shape, dtype=self.global_data.dtype)
        self.global_data.rmsimg = np.zeros(self.global_data.data_pix.shape, dtype=self.global_data.dtype)
        self.global_data.pixarea = img.pixarea
        self.global_data.dcurve = None

        if do_curve:
            self.log.info("Calculating curvature")
            # calculate curvature but store it as -1,0,+1
            dcurve = np.zeros(self.global_data.data_pix.shape, dtype=np.int8)
            peaks = scipy.ndimage.filters.maximum_filter(self.global_data.data_pix, size=3)
            troughs = scipy.ndimage.filters.minimum_filter(self.global_data.data_pix, size=3)
            pmask = np.where(self.global_data.data_pix == peaks)
            tmask = np.where(self.global_data.data_pix == troughs)
            dcurve[pmask] = -1
            dcurve[tmask] = 1
            self.global_data.dcurve = dcurve

        # if either of rms or bkg images are not supplied then calculate them both
        if not (rmsin and bkgin):
            if verb:
                self.log.info("Calculating background and rms data")
            self._make_bkg_rms(mesh_size=20, forced_rms=rms, forced_bkg=bkg, cores=cores)

        # replace the calculated images with input versions, if the user has supplied them.
        if bkgin:
            if verb:
                self.log.info("Loading background data from file {0}".format(bkgin))
            self.global_data.bkgimg = self._load_aux_image(img, bkgin)
        if rmsin:
            if verb:
                self.log.info("Loading rms data from file {0}".format(rmsin))
            self.global_data.rmsimg = self._load_aux_image(img, rmsin)

        # subtract the background image from the data image and save
        if verb and debug:
            self.log.debug("Data max is {0}".format(img.get_pixels()[np.isfinite(img.get_pixels())].max()))
            self.log.debug("Doing background subtraction")
        img.set_pixels(img.get_pixels() - self.global_data.bkgimg)
        self.global_data.data_pix = img.get_pixels()
        if verb and debug:
            self.log.debug("Data max is {0}".format(img.get_pixels()[np.isfinite(img.get_pixels())].max()))

        self.global_data.blank = blank
        self.global_data.docov = docov

        # Default to false until I can verify that this is working
        self.global_data.dobias = False

        # check if the WCS is galactic
        if 'lon' in self.global_data.img._header['CTYPE1'].lower():
            self.log.info("Galactic coordinates detected and noted")
            SimpleSource.galactic = True
        return