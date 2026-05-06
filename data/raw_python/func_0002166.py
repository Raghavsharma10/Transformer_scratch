def time(self, time):
        """Add a request for a specific time to the query.

        This modifies the query in-place, but returns `self` so that multiple queries
        can be chained together on one line.

        This replaces any existing temporal queries that have been set.

        Parameters
        ----------
        time : datetime.datetime
            The time to request

        Returns
        -------
        self : DataQuery
            Returns self for chaining calls

        """
        self._set_query(self.time_query, time=self._format_time(time))
        return self