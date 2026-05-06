def filter_time_nearest(self, time, regex=None):
        """Filter keys for an item closest to the desired time.

        Loops over all keys in the collection and uses `regex` to extract and build
        `datetime`s. The collection of `datetime`s is compared to `start` and the value that
        has a `datetime` closest to that requested is returned.If none of the keys in the
        collection match the regex, indicating that the keys are not date/time-based,
        a ``ValueError`` is raised.

        Parameters
        ----------
        time : ``datetime.datetime``
            The desired time
        regex : str, optional
            The regular expression to use to extract date/time information from the key. If
            given, this should contain named groups: 'year', 'month', 'day', 'hour', 'minute',
            'second', and 'microsecond', as appropriate. When a match is found, any of those
            groups missing from the pattern will be assigned a value of 0. The default pattern
            looks for patterns like: 20171118_2356.

        Returns
        -------
            The value with a time closest to that desired

        """
        return min(self._get_datasets_with_times(regex),
                   key=lambda i: abs((i[0] - time).total_seconds()))[-1]