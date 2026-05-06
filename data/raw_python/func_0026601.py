def qd2apex(self, qlat, qlon, height):
        """Converts quasi-dipole to modified apex coordinates.

        Parameters
        ==========
        qlat : array_like
            Quasi-dipole latitude
        qlon : array_like
            Quasi-dipole longitude
        height : array_like
            Altitude in km

        Returns
        =======
        alat : ndarray or float
            Modified apex latitude
        alon : ndarray or float
            Modified apex longitude

        Raises
        ======
        ApexHeightError
            if apex height < reference height

        """

        alat, alon = self._qd2apex(qlat, qlon, height)

        # if array is returned, the dtype is object, so convert to float
        return np.float64(alat), np.float64(alon)