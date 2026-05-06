def apex2qd(self, alat, alon, height):
        """Converts modified apex to quasi-dipole coordinates.

        Parameters
        ==========
        alat : array_like
            Modified apex latitude
        alon : array_like
            Modified apex longitude
        height : array_like
            Altitude in km

        Returns
        =======
        qlat : ndarray or float
            Quasi-dipole latitude
        qlon : ndarray or float
            Quasi-dipole longitude

        Raises
        ======
        ApexHeightError
            if `height` > apex height

        """

        qlat, qlon = self._apex2qd(alat, alon, height)

        # if array is returned, the dtype is object, so convert to float
        return np.float64(qlat), np.float64(qlon)