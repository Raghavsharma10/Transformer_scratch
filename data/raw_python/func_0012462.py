def _get_cache_dates(self):
        """
        Get s list of dates (:py:class:`datetime.datetime`) present in cache,
        beginning with the longest contiguous set of dates that isn't missing
        more than one date in series.

        :return: list of datetime objects for contiguous dates in cache
        :rtype: ``list``
        """
        all_dates = self.cache.get_dates_for_project(self.project_name)
        dates = []
        last_date = None
        for val in sorted(all_dates):
            if last_date is None:
                last_date = val
                continue
            if val - last_date > timedelta(hours=48):
                # reset dates to start from here
                logger.warning("Last cache date was %s, current date is %s; "
                               "delta is too large. Starting cache date series "
                               "at current date.", last_date, val)
                dates = []
            last_date = val
            dates.append(val)
        # find the first download record, and only look at dates after that
        for idx, cache_date in enumerate(dates):
            data = self._cache_get(cache_date)
            if not self._is_empty_cache_record(data):
                logger.debug("First cache date with data: %s", cache_date)
                return dates[idx:]
        return dates