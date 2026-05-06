def mlt2mlon(self, mlt, datetime, ssheight=50*6371):
        """Computes the magnetic longitude at the specified magnetic local time
        and UT.

        Parameters
        ==========
        mlt : array_like
            Magnetic local time
        datetime : :class:`datetime.datetime`
            Date and time
        ssheight : float, optional
            Altitude in km to use for converting the subsolar point from
            geographic to magnetic coordinates. A high altitude is used
            to ensure the subsolar point is mapped to high latitudes, which
            prevents the South-Atlantic Anomaly (SAA) from influencing the MLT.

        Returns
        =======
        mlon : ndarray or float
            Magnetic longitude [0, 360) (apex and quasi-dipole longitude are
            always equal)

        Notes
        =====
        To compute the magnetic longitude, we find the apex longitude of the
        subsolar point at the given time. Then the magnetic longitude of the
        given point will be computed from the separation in magnetic local time
        from this point (1 hour = 15 degrees).
        """

        ssglat, ssglon = helpers.subsol(datetime)
        ssalat, ssalon = self.geo2apex(ssglat, ssglon, ssheight)

        # np.float64 will ensure lists are converted to arrays
        return (15*np.float64(mlt) - 180 + ssalon + 360) % 360