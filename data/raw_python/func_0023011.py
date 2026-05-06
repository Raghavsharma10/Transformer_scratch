def _to_fits(self, uniq, optional_kw_dict=None):
        """
        Serializes a MOC to the FITS format.

        Parameters
        ----------
        uniq : `numpy.ndarray`
            The array of HEALPix cells representing the MOC to serialize.
        optional_kw_dict : dict
            Optional keywords arguments added to the FITS header.

        Returns
        -------
        thdulist : `astropy.io.fits.HDUList`
            The list of HDU tables.
        """
        depth = self.max_order
        if depth <= 13:
            fits_format = '1J'
        else:
            fits_format = '1K'

        tbhdu = fits.BinTableHDU.from_columns(
            fits.ColDefs([
                fits.Column(name='UNIQ', format=fits_format, array=uniq)
            ]))
        tbhdu.header['PIXTYPE'] = 'HEALPIX'
        tbhdu.header['ORDERING'] = 'NUNIQ'
        tbhdu.header.update(self._fits_header_keywords)
        tbhdu.header['MOCORDER'] = depth
        tbhdu.header['MOCTOOL'] = 'MOCPy'
        if optional_kw_dict:
            for key in optional_kw_dict:
                tbhdu.header[key] = optional_kw_dict[key]

        thdulist = fits.HDUList([fits.PrimaryHDU(), tbhdu])
        return thdulist