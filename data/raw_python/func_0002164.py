def lonlat_box(self, west, east, south, north):
        """Add a latitude/longitude bounding box to the query.

        This adds a request for a spatial bounding box, bounded by ('north', 'south')
        for latitude and ('east', 'west') for the longitude. This modifies the query
        in-place, but returns `self` so that multiple queries can be chained together
        on one line.

        This replaces any existing spatial queries that have been set.

        Parameters
        ----------
        west: float
            The bounding longitude to the west, in degrees east of the prime meridian
        east : float
            The bounding longitude to the east, in degrees east of the prime meridian
        south : float
            The bounding latitude to the south, in degrees north of the equator
        north : float
            The bounding latitude to the north, in degrees north of the equator

        Returns
        -------
        self : DataQuery
            Returns self for chaining calls

        """
        self._set_query(self.spatial_query, west=west, east=east, south=south,
                        north=north)
        return self