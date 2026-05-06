def stations(self, *stns):
        """Specify one or more stations for the query.

        This modifies the query in-place, but returns `self` so that multiple
        queries can be chained together on one line.

        This replaces any existing spatial queries that have been set.

        Parameters
        ----------
        stns : one or more strings
            One or more names of variables to request

        Returns
        -------
        self : RadarQuery
            Returns self for chaining calls

        """
        self._set_query(self.spatial_query, stn=stns)
        return self