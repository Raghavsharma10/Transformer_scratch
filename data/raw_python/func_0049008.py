def limit_result_set(self, start, end):
        """By default, searches return all matching results.

        This method restricts the number of results by setting the start
        and end of the result set, starting from 1. The starting and
        ending results can be used for paging results when a certain
        ordering is requested. The ending position must be greater than
        the starting position.

        arg:    start (cardinal): the start of the result set
        arg:    end (cardinal): the end of the result set
        raise:  InvalidArgument - ``end`` is less than or equal to
                ``start``
        *compliance: mandatory -- This method must be implemented.*

        """
        if not isinstance(start, int) or not isinstance(end, int):
            raise errors.InvalidArgument('start and end arguments must be integers.')
        if end <= start:
            raise errors.InvalidArgument('End must be greater than start.')

        # because Python is 0 indexed
        # Spec says that passing in (1, 25) should include 25 entries (1 - 25)
        # Python indices 0 - 24
        # Python [#:##] stops before the last index, but does not include it
        self._limit_result_set_start = start - 1
        self._limit_result_set_end = end