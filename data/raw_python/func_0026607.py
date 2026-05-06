def basevectors_qd(self, lat, lon, height, coords='geo', precision=1e-10):
        """Returns quasi-dipole base vectors f1 and f2 at the specified
        coordinates.

        The vectors are described by Richmond [1995] [2]_ and
        Emmert et al. [2010] [3]_.  The vector components are geodetic east and
        north.

        Parameters
        ==========
        lat : (N,) array_like or float
            Latitude
        lon : (N,) array_like or float
            Longitude
        height : (N,) array_like or float
            Altitude in km
        coords : {'geo', 'apex', 'qd'}, optional
            Input coordinate system
        precision : float, optional
            Precision of output (degrees) when converting to geo. A negative
            value of this argument produces a low-precision calculation of
            geodetic lat/lon based only on their spherical harmonic
            representation.
            A positive value causes the underlying Fortran routine to iterate
            until feeding the output geo lat/lon into geo2qd (APXG2Q) reproduces
            the input QD lat/lon to within the specified precision (all
            coordinates being converted to geo are converted to QD first and
            passed through APXG2Q).

        Returns
        =======
        f1 : (2, N) or (2,) ndarray
        f2 : (2, N) or (2,) ndarray

        References
        ==========
        .. [2] Richmond, A. D. (1995), Ionospheric Electrodynamics Using
               Magnetic Apex Coordinates, Journal of geomagnetism and
               geoelectricity, 47(2), 191–212, :doi:`10.5636/jgg.47.191`.

        .. [3] Emmert, J. T., A. D. Richmond, and D. P. Drob (2010),
               A computationally compact representation of Magnetic-Apex
               and Quasi-Dipole coordinates with smooth base vectors,
               J. Geophys. Res., 115(A8), A08322, :doi:`10.1029/2010JA015326`.

        """

        glat, glon = self.convert(lat, lon, coords, 'geo', height=height,
                                  precision=precision)

        f1, f2 = self._basevec(glat, glon, height)

        # if inputs are not scalar, each vector is an array of arrays,
        # so reshape to a single array
        if f1.dtype == object:
            f1 = np.vstack(f1).T
            f2 = np.vstack(f2).T

        return f1, f2