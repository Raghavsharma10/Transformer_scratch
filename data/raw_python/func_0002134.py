def filter_time_range(self, start, end, regex=None):
        """Filter keys for all items within the desired time range.

        Loops over all keys in the collection and uses `regex` to extract and build
        `datetime`s. From the collection of `datetime`s, all values within `start` and `end`
        (inclusive) are returned. If none of the keys in the collection match the regex,
        indicating that the keys are not date/time-based, a ``ValueError`` is raised.

        Parameters
        ----------
        start : ``datetime.datetime``
            The start of the desired time range, inclusive
        end : ``datetime.datetime``
            The end of the desired time range, inclusive
        regex : str, optional
            The regular expression to use to extract date/time information from the key. If
            given, this should contain named groups: 'year', 'month', 'day', 'hour', 'minute',
            'second', and 'microsecond', as appropriate. When a match is found, any of those
            groups missing from the pattern will be assigned a value of 0. The default pattern
            looks for patterns like: 20171118_2356.

        Returns
        -------
            All values corresponding to times within the specified range

        """
        return [item[-1] for item in self._get_datasets_with_times(regex)
                if start <= item[0] <= end]