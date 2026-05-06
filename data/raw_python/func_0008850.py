def save_background_files(self, image_filename, hdu_index=0, bkgin=None, rmsin=None, beam=None, rms=None, bkg=None, cores=1,
                              outbase=None):
        """
        Generate and save the background and RMS maps as FITS files.
        They are saved in the current directly as aegean-background.fits and aegean-rms.fits.

        Parameters
        ----------
        image_filename : str or HDUList
            Input image.

        hdu_index : int
            If fits file has more than one hdu, it can be specified here.
            Default = 0.

        bkgin, rmsin : str or HDUList
            Background and noise image filename or HDUList

        beam : :class:`AegeanTools.fits_image.Beam`
            Beam object representing the synthsized beam. Will replace what is in the FITS header.


        rms, bkg : float
            A float that represents a constant rms/bkg level for the entire image.
            Default = None, which causes the rms/bkg to be loaded or calculated.

        cores : int
            Number of cores to use if different from what is autodetected.

        outbase : str
            Basename for output files.

        """

        self.log.info("Saving background / RMS maps")
        # load image, and load/create background/rms images
        self.load_globals(image_filename, hdu_index=hdu_index, bkgin=bkgin, rmsin=rmsin, beam=beam, verb=True, rms=rms, bkg=bkg,
                          cores=cores, do_curve=True)
        img = self.global_data.img
        bkgimg, rmsimg = self.global_data.bkgimg, self.global_data.rmsimg
        curve = np.array(self.global_data.dcurve, dtype=bkgimg.dtype)
        # mask these arrays have the same mask the same as the data
        mask = np.where(np.isnan(self.global_data.data_pix))
        bkgimg[mask] = np.NaN
        rmsimg[mask] = np.NaN
        curve[mask] = np.NaN

        # Generate the new FITS files by copying the existing HDU and assigning new data.
        # This gives the new files the same WCS projection and other header fields.
        new_hdu = img.hdu
        # Set the ORIGIN to indicate Aegean made this file
        new_hdu.header["ORIGIN"] = "Aegean {0}-({1})".format(__version__, __date__)
        for c in ['CRPIX3', 'CRPIX4', 'CDELT3', 'CDELT4', 'CRVAL3', 'CRVAL4', 'CTYPE3', 'CTYPE4']:
            if c in new_hdu.header:
                del new_hdu.header[c]

        if outbase is None:
            outbase, _ = os.path.splitext(os.path.basename(image_filename))
        noise_out = outbase + '_rms.fits'
        background_out = outbase + '_bkg.fits'
        curve_out = outbase + '_crv.fits'
        snr_out = outbase + '_snr.fits'

        new_hdu.data = bkgimg
        new_hdu.writeto(background_out, overwrite=True)
        self.log.info("Wrote {0}".format(background_out))

        new_hdu.data = rmsimg
        new_hdu.writeto(noise_out, overwrite=True)
        self.log.info("Wrote {0}".format(noise_out))

        new_hdu.data = curve
        new_hdu.writeto(curve_out, overwrite=True)
        self.log.info("Wrote {0}".format(curve_out))

        new_hdu.data = self.global_data.data_pix / rmsimg
        new_hdu.writeto(snr_out, overwrite=True)
        self.log.info("Wrote {0}".format(snr_out))
        return