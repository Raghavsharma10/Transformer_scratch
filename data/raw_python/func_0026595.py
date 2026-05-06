def geo2apex(self, glat, glon, height):
        """Converts geodetic to modified apex coordinates.

        Parameters
        ==========
        glat : array_like
            Geodetic latitude
        glon : array_like
            Geodetic longitude
        height : array_like
            Altitude in km

        Returns
        =======
        alat : ndarray or float
            Modified apex latitude
        alon : ndarray or float
            Modified apex longitude

        """

        glat = helpers.checklat(glat, name='glat')

        alat, alon = self._geo2apex(glat, glon, height)

        if np.any(np.float64(alat) == -9999):
            warnings.warn('Apex latitude set to -9999 where undefined '
                          '(apex height may be < reference height)')

        # if array is returned, dtype is object, so convert to float
        return np.float64(alat), np.float64(alon)