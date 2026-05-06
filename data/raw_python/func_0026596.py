def apex2geo(self, alat, alon, height, precision=1e-10):
        """Converts modified apex to geodetic coordinates.

        Parameters
        ==========
        alat : array_like
            Modified apex latitude
        alon : array_like
            Modified apex longitude
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

        alat = helpers.checklat(alat, name='alat')

        qlat, qlon = self.apex2qd(alat, alon, height=height)
        glat, glon, error = self.qd2geo(qlat, qlon, height, precision=precision)

        return glat, glon, error