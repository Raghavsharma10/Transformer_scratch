def _load_aux_image(self, image, auxfile):
        """
        Load a fits file (bkg/rms/curve) and make sure that
        it is the same shape as the main image.

        Parameters
        ----------
        image : :class:`AegeanTools.fits_image.FitsImage`
            The main image that has already been loaded.

        auxfile : str or HDUList
            The auxiliary file to be loaded.

        Returns
        -------
        aux : :class:`AegeanTools.fits_image.FitsImage`
            The loaded image.
        """
        auximg = FitsImage(auxfile, beam=self.global_data.beam).get_pixels()
        if auximg.shape != image.get_pixels().shape:
            self.log.error("file {0} is not the same size as the image map".format(auxfile))
            self.log.error("{0}= {1}, image = {2}".format(auxfile, auximg.shape, image.get_pixels().shape))
            sys.exit(1)
        return auximg