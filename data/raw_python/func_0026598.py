def qd2geo(self, qlat, qlon, height, precision=1e-10):
        """Converts quasi-dipole to geodetic coordinates.

        Parameters
        ==========
        qlat : array_like
            Quasi-dipole latitude
        qlon : array_like
            Quasi-dipole longitude
        height : array_like
            Altitude in km
        precision : float, optional
            Precision of output (degrees). A negative value of this argument
            produces a low-precision calculation of geodetic lat/lon based only
            on their spherical harmonic representation. A positive value causes
            the underlying Fortran routine to iterate until feeding the output
            geo lat/lon into geo2qd (APXG2Q) reproduces the input QD lat/lon to
            within the specified precision.

        Returns
        =======
        glat : ndarray or float
            Geodetic latitude
        glon : ndarray or float
            Geodetic longitude
        error : ndarray or float
            The angular difference (degrees) between the input QD coordinates
            and the qlat/qlon produced by feeding the output glat and glon
            into geo2qd (APXG2Q)

        """

        qlat = helpers.checklat(qlat, name='qlat')

        glat, glon, error = self._qd2geo(qlat, qlon, height, precision)

        # if array is returned, dtype is object, so convert to float
        return np.float64(glat), np.float64(glon), np.float64(error)