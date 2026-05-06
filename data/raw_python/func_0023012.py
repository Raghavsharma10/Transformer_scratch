def serialize(self, format='fits', optional_kw_dict=None):
        """
        Serializes the MOC into a specific format.

        Possible formats are FITS, JSON and STRING

        Parameters
        ----------
        format : str
            'fits' by default. The other possible choice is 'json' or 'str'.
        optional_kw_dict : dict
            Optional keywords arguments added to the FITS header. Only used if ``format`` equals to 'fits'.

        Returns
        -------
        result : `astropy.io.fits.HDUList` or JSON dictionary
            The result of the serialization.
        """
        formats = ('fits', 'json', 'str')
        if format not in formats:
            raise ValueError('format should be one of %s' % (str(formats)))

        uniq_l = []
        for uniq in self._uniq_pixels_iterator():
            uniq_l.append(uniq)

        uniq = np.array(uniq_l)

        if format == 'fits':
            result = self._to_fits(uniq=uniq,
                                   optional_kw_dict=optional_kw_dict)
        elif format == 'str':
            result = self.__class__._to_str(uniq=uniq)
        else:
            # json format serialization
            result = self.__class__._to_json(uniq=uniq)

        return result