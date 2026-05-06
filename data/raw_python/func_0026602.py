def mlon2mlt(self, mlon, datetime, ssheight=50*6371):
        """Computes the magnetic local time at the specified magnetic longitude
        and UT.

        Parameters
        ==========
        mlon : array_like
            Magnetic longitude (apex and quasi-dipole longitude are always 
            equal)
        datetime : :class:`datetime.datetime`
            Date and time
        ssheight : float, optional
            Altitude in km to use for converting the subsolar point from
            geographic to magnetic coordinates. A high altitude is used
            to ensure the subsolar point is mapped to high latitudes, which
            prevents the South-Atlantic Anomaly (SAA) from influencing the MLT.

        Returns
        =======
        mlt : ndarray or float
            Magnetic local time [0, 24)

        Notes
        =====
        To compute the MLT, we find the apex longitude of the subsolar point at
        the given time. Then the MLT of the given point will be computed from
        the separation in magnetic longitude from this point (1 hour = 15
        degrees).

        """
        ssglat, ssglon = helpers.subsol(datetime)
        ssalat, ssalon = self.geo2apex(ssglat, ssglon, ssheight)

        # np.float64 will ensure lists are converted to arrays
        return (180 + np.float64(mlon) - ssalon)/15 % 24