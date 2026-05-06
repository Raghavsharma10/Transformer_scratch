def get_lonlat(self):
        """
        Calculate longitude-latitude grid for a specified resolution and
        configuration / ordering.

        Parameters
        ----------
        rlon, rlat : float
            Resolution (in degrees) of longitude and latitude grids.
        halfpolar : bool (default=True)
            Polar grid boxes span half of rlat relative to the other grid cells.
        center180 : bool (default=True)
            Longitude grid should be centered at 180 degrees.

        """

        rlon, rlat = self.resolution

        # Compute number of grid cells in each direction
        Nlon = int(360. / rlon)
        Nlat = int(180. / rlat) + self.halfpolar

        # Compute grid cell edges
        elon = np.arange(Nlon + 1) * rlon - np.array(180.)
        elon -= rlon / 2. * self.center180
        elat = np.arange(Nlat + 1) * rlat - np.array(90.)
        elat -= rlat / 2. * self.halfpolar
        elat[0] = -90.
        elat[-1] = 90.

        # Compute grid cell centers
        clon = (elon - (rlon / 2.))[1:]
        clat = np.arange(Nlat) * rlat - np.array(90.)

        # Fix grid boundaries if halfpolar
        if self.halfpolar:
            clat[0] = (elat[0] + elat[1]) / 2.
            clat[-1] = -clat[0]
        else:
            clat += (elat[1] - elat[0]) / 2.

        return {
            "lon_centers": clon, "lat_centers": clat,
            "lon_edges": elon, "lat_edges": elat
        }