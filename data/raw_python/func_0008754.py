def radec2sky(ra, dec):
        """
        Convert [ra], [dec] to [(ra[0], dec[0]),....]
        and also  ra,dec to [(ra,dec)] if ra/dec are not iterable

        Parameters
        ----------
        ra, dec : float or iterable
            Sky coordinates

        Returns
        -------
        sky : numpy.array
            array of (ra,dec) coordinates.
        """
        try:
            sky = np.array(list(zip(ra, dec)))
        except TypeError:
            sky = np.array([(ra, dec)])
        return sky