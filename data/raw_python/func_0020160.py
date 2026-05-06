def load_fits(self):
        '''
        Load the FITS file from disk and populate the
        class instance with its data.

        '''

        log.info("Loading FITS file for %d." % (self.ID))
        with pyfits.open(self.fitsfile) as f:

            # Params and long cadence data
            self.loaded = True
            self.is_parent = False
            try:
                self.X1N = f[2].data['X1N']
            except KeyError:
                self.X1N = None
            self.aperture = f[3].data
            self.aperture_name = f[1].header['APNAME']
            try:
                self.bkg = f[1].data['BKG']
            except KeyError:
                self.bkg = 0.
            self.bpad = f[1].header['BPAD']
            self.cbv_minstars = []
            self.cbv_num = f[1].header.get('CBVNUM', 1)
            self.cbv_niter = f[1].header['CBVNITER']
            self.cbv_win = f[1].header['CBVWIN']
            self.cbv_order = f[1].header['CBVORD']
            self.cadn = f[1].data['CADN']
            self.cdivs = f[1].header['CDIVS']
            self.cdpp = f[1].header['CDPP']
            self.cdppr = f[1].header['CDPPR']
            self.cdppv = f[1].header['CDPPV']
            self.cdppg = f[1].header['CDPPG']
            self.cv_min = f[1].header['CVMIN']
            self.fpix = f[2].data['FPIX']
            self.pixel_images = [f[4].data['STAMP1'],
                                 f[4].data['STAMP2'], f[4].data['STAMP3']]
            self.fraw = f[1].data['FRAW']
            self.fraw_err = f[1].data['FRAW_ERR']
            self.giter = f[1].header['GITER']
            self.gmaxf = f[1].header.get('GMAXF', 200)
            self.gp_factor = f[1].header['GPFACTOR']
            try:
                self.hires = f[5].data
            except:
                self.hires = None
            self.kernel_params = np.array([f[1].header['GPWHITE'],
                                           f[1].header['GPRED'],
                                           f[1].header['GPTAU']])
            try:
                self.kernel = f[1].header['KERNEL']
                self.kernel_params = np.append(
                    self.kernel_params,
                    [f[1].header['GPGAMMA'],
                     f[1].header['GPPER']])
            except KeyError:
                self.kernel = 'Basic'
            self.pld_order = f[1].header['PLDORDER']
            self.lam_idx = self.pld_order
            self.leps = f[1].header['LEPS']
            self.mag = f[0].header['KEPMAG']
            self.max_pixels = f[1].header['MAXPIX']
            self.model = self.fraw - f[1].data['FLUX']
            self.nearby = []
            for i in range(99):
                try:
                    ID = f[1].header['NRBY%02dID' % (i + 1)]
                    x = f[1].header['NRBY%02dX' % (i + 1)]
                    y = f[1].header['NRBY%02dY' % (i + 1)]
                    mag = f[1].header['NRBY%02dM' % (i + 1)]
                    x0 = f[1].header['NRBY%02dX0' % (i + 1)]
                    y0 = f[1].header['NRBY%02dY0' % (i + 1)]
                    self.nearby.append(
                        {'ID': ID, 'x': x, 'y': y,
                         'mag': mag, 'x0': x0, 'y0': y0})
                except KeyError:
                    break
            self.neighbors = []
            for c in range(99):
                try:
                    self.neighbors.append(f[1].header['NEIGH%02d' % (c + 1)])
                except KeyError:
                    break
            self.oiter = f[1].header['OITER']
            self.optimize_gp = f[1].header['OPTGP']
            self.osigma = f[1].header['OSIGMA']
            self.planets = []
            for i in range(99):
                try:
                    t0 = f[1].header['P%02dT0' % (i + 1)]
                    per = f[1].header['P%02dPER' % (i + 1)]
                    dur = f[1].header['P%02dDUR' % (i + 1)]
                    self.planets.append((t0, per, dur))
                except KeyError:
                    break
            self.quality = f[1].data['QUALITY']
            self.saturated = f[1].header['SATUR']
            self.saturation_tolerance = f[1].header['SATTOL']
            self.time = f[1].data['TIME']
            self._norm = np.array(self.fraw)

            # Chunk arrays
            self.breakpoints = []
            self.cdpp_arr = []
            self.cdppv_arr = []
            self.cdppr_arr = []
            for c in range(99):
                try:
                    self.breakpoints.append(f[1].header['BRKPT%02d' % (c + 1)])
                    self.cdpp_arr.append(f[1].header['CDPP%02d' % (c + 1)])
                    self.cdppr_arr.append(f[1].header['CDPPR%02d' % (c + 1)])
                    self.cdppv_arr.append(f[1].header['CDPPV%02d' % (c + 1)])
                except KeyError:
                    break
            self.lam = [[f[1].header['LAMB%02d%02d' % (c + 1, o + 1)]
                        for o in range(self.pld_order)]
                        for c in range(len(self.breakpoints))]
            if self.model_name == 'iPLD':
                self.reclam = [[f[1].header['RECL%02d%02d' % (c + 1, o + 1)]
                               for o in range(self.pld_order)]
                               for c in range(len(self.breakpoints))]

            # Masks
            self.badmask = np.where(self.quality & 2 ** (QUALITY_BAD - 1))[0]
            self.nanmask = np.where(self.quality & 2 ** (QUALITY_NAN - 1))[0]
            self.outmask = np.where(self.quality & 2 ** (QUALITY_OUT - 1))[0]
            self.recmask = np.where(self.quality & 2 ** (QUALITY_REC - 1))[0]
            self.transitmask = np.where(
                self.quality & 2 ** (QUALITY_TRN - 1))[0]

            # CBVs
            self.XCBV = np.empty((len(self.time), 0))
            for i in range(99):
                try:
                    self.XCBV = np.hstack(
                        [self.XCBV,
                         f[1].data['CBV%02d' % (i + 1)].reshape(-1, 1)])
                except KeyError:
                    break

        # These are not stored in the fits file; we don't need them
        self.saturated_aperture_name = None
        self.apertures = None
        self.Xpos = None
        self.Ypos = None
        self.fpix_err = None
        self.parent_model = None
        self.lambda_arr = None
        self.meta = None
        self._transit_model = None
        self.transit_depth = None