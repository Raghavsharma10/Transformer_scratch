def basevectors_apex(self, lat, lon, height, coords='geo', precision=1e-10):
        """Returns base vectors in quasi-dipole and apex coordinates.

        The vectors are described by Richmond [1995] [4]_ and
        Emmert et al. [2010] [5]_.  The vector components are geodetic east,
        north, and up (only east and north for `f1` and `f2`).

        Parameters
        ==========
        lat, lon : (N,) array_like or float
            Latitude
        lat : (N,) array_like or float
            Longitude
        height : (N,) array_like or float
            Altitude in km
        coords : {'geo', 'apex', 'qd'}, optional
            Input coordinate system
        return_all : bool, optional
            Will also return f3, g1, g2, and g3, and f1 and f2 have 3 components
            (the last component is zero). Requires `lat`, `lon`, and `height`
            to be broadcast to 1D (at least one of the parameters must be 1D
            and the other two parameters must be 1D or 0D).
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
        f1, f2 : (2, N) or (2,) ndarray
        f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 : (3, N) or (3,) ndarray

        Note
        ====
        `f3`, `g1`, `g2`, and `g3` are not part of the Fortran code
        by Emmert et al. [2010] [5]_. They are calculated by this
        Python library according to the following equations in
        Richmond [1995] [4]_:

        * `g1`: Eqn. 6.3
        * `g2`: Eqn. 6.4
        * `g3`: Eqn. 6.5
        * `f3`: Eqn. 6.8

        References
        ==========

        .. [4] Richmond, A. D. (1995), Ionospheric Electrodynamics Using
               Magnetic Apex Coordinates, Journal of geomagnetism and
               geoelectricity, 47(2), 191–212, :doi:`10.5636/jgg.47.191`.

        .. [5] Emmert, J. T., A. D. Richmond, and D. P. Drob (2010),
               A computationally compact representation of Magnetic-Apex
               and Quasi-Dipole coordinates with smooth base vectors,
               J. Geophys. Res., 115(A8), A08322, :doi:`10.1029/2010JA015326`.

        """

        glat, glon = self.convert(lat, lon, coords, 'geo', height=height,
                                  precision=precision)

        returnvals = self._geo2apexall(glat, glon, height)
        qlat = np.float64(returnvals[0])
        alat = np.float64(returnvals[2])
        f1, f2 = returnvals[4:6]
        d1, d2, d3 = returnvals[7:10]
        e1, e2, e3 = returnvals[11:14]

        # if inputs are not scalar, each vector is an array of arrays,
        # so reshape to a single array
        if f1.dtype == object:
            f1 = np.vstack(f1).T
            f2 = np.vstack(f2).T
            d1 = np.vstack(d1).T
            d2 = np.vstack(d2).T
            d3 = np.vstack(d3).T
            e1 = np.vstack(e1).T
            e2 = np.vstack(e2).T
            e3 = np.vstack(e3).T

        # make sure arrays are 2D
        f1 = f1.reshape((2, f1.size//2))
        f2 = f2.reshape((2, f2.size//2))
        d1 = d1.reshape((3, d1.size//3))
        d2 = d2.reshape((3, d2.size//3))
        d3 = d3.reshape((3, d3.size//3))
        e1 = e1.reshape((3, e1.size//3))
        e2 = e2.reshape((3, e2.size//3))
        e3 = e3.reshape((3, e3.size//3))

        # compute f3, g1, g2, g3
        F1 = np.vstack((f1, np.zeros_like(f1[0])))
        F2 = np.vstack((f2, np.zeros_like(f2[0])))
        F = np.cross(F1.T, F2.T).T[-1]
        cosI = helpers.getcosIm(alat)
        k = np.array([0, 0, 1], dtype=np.float64).reshape((3, 1))
        g1 = ((self.RE + np.float64(height)) / (self.RE + self.refh))**(3/2) \
             * d1 / F
        g2 = -1.0 / (2.0 * F * np.tan(np.radians(qlat))) * \
             (k + ((self.RE + np.float64(height)) / (self.RE + self.refh))
              * d2 / cosI)
        g3 = k*F
        f3 = np.cross(g1.T, g2.T).T

        if np.any(alat == -9999):
            warnings.warn(('Base vectors g, d, e, and f3 set to -9999 where '
                           'apex latitude is undefined (apex height may be < '
                           'reference height)'))
            f3 = np.where(alat == -9999, -9999, f3)
            g1 = np.where(alat == -9999, -9999, g1)
            g2 = np.where(alat == -9999, -9999, g2)
            g3 = np.where(alat == -9999, -9999, g3)
            d1 = np.where(alat == -9999, -9999, d1)
            d2 = np.where(alat == -9999, -9999, d2)
            d3 = np.where(alat == -9999, -9999, d3)
            e1 = np.where(alat == -9999, -9999, e1)
            e2 = np.where(alat == -9999, -9999, e2)
            e3 = np.where(alat == -9999, -9999, e3)

        return tuple(np.squeeze(x) for x in
                     [f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3])