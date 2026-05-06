def lonlat_point(self, lon, lat):
        """Add a latitude/longitude point to the query.

        This adds a request for a (`lon`, `lat`) point. This modifies the query
        in-place, but returns `self` so that multiple queries can be chained together on
        one line.

        This replaces any existing spatial queries that have been set.

        Parameters
        ----------
        lon: float
            The longitude to request
        lat : float
            The latitude to request

        Returns
        -------
        self : DataQuery
            Returns self for chaining calls

        """
        self._set_query(self.spatial_query, longitude=lon, latitude=lat)
        return self