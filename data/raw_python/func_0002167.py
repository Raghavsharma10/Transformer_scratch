def time_range(self, start, end):
        """Add a request for a time range to the query.

        This modifies the query in-place, but returns `self` so that multiple queries
        can be chained together on one line.

        This replaces any existing temporal queries that have been set.

        Parameters
        ----------
        start : datetime.datetime
            The start of the requested time range
        end : datetime.datetime
            The end of the requested time range

        Returns
        -------
        self : DataQuery
            Returns self for chaining calls

        """
        self._set_query(self.time_query, time_start=self._format_time(start),
                        time_end=self._format_time(end))
        return self