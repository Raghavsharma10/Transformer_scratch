def map_to_height(self, glat, glon, height, newheight, conjugate=False,
                      precision=1e-10):
        """Performs mapping of points along the magnetic field to the closest
        or conjugate hemisphere.

        Parameters
        ==========
        glat : array_like
            Geodetic latitude
        glon : array_like
            Geodetic longitude
        height : array_like
            Source altitude in km
        newheight : array_like
            Destination altitude in km
        conjugate : bool, optional
            Map to `newheight` in the conjugate hemisphere instead of the
            closest hemisphere
        precision : float, optional
            Precision of output (degrees). A negative value of this argument
            produces a low-precision calculation of geodetic lat/lon based only
            on their spherical harmonic representation. A positive value causes
            the underlying Fortran routine to iterate until feeding the output
            geo lat/lon into geo2qd (APXG2Q) reproduces the input QD lat/lon to
            within the specified precision.

        Returns
        =======
        newglat : ndarray or float
            Geodetic latitude of mapped point
        newglon : ndarray or float
            Geodetic longitude of mapped point
        error : ndarray or float
            The angular difference (degrees) between the input QD coordinates
            and the qlat/qlon produced by feeding the output glat and glon
            into geo2qd (APXG2Q)

        Notes
        =====
        The mapping is done by converting glat/glon/height to modified apex
        lat/lon, and converting back to geographic using newheight (if
        conjugate, use negative apex latitude when converting back)

        """

        alat, alon = self.geo2apex(glat, glon, height)
        if conjugate:
            alat = -alat
        try:
            newglat, newglon, error = self.apex2geo(alat, alon, newheight,
                                                    precision=precision)
        except ApexHeightError:
            raise ApexHeightError("newheight is > apex height")

        return newglat, newglon, error