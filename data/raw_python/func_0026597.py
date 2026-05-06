def geo2qd(self, glat, glon, height):
        """Converts geodetic to quasi-dipole coordinates.

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
        qlat : ndarray or float
            Quasi-dipole latitude
        qlon : ndarray or float
            Quasi-dipole longitude

        """

        glat = helpers.checklat(glat, name='glat')

        qlat, qlon = self._geo2qd(glat, glon, height)

        # if array is returned, dtype is object, so convert to float
        return np.float64(qlat), np.float64(qlon)