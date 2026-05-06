def _downloads_for_num_days(self, num_days):
        """
        Given a number of days of historical data to look at (starting with
        today and working backwards), return the total number of downloads
        for that time range, and the number of days of data we had (in cases
        where we had less data than requested).

        :param num_days: number of days of data to look at
        :type num_days: int
        :return: 2-tuple of (download total, number of days of data)
        :rtype: tuple
        """
        logger.debug("Getting download total for last %d days", num_days)
        dates = self.cache_dates
        logger.debug("Cache has %d days of data", len(dates))
        if len(dates) > num_days:
            dates = dates[(-1 * num_days):]
        logger.debug("Looking at last %d days of data", len(dates))
        dl_sum = 0
        for cache_date in dates:
            data = self._cache_get(cache_date)
            dl_sum += sum(data['by_version'].values())
        logger.debug("Sum of download counts: %d", dl_sum)
        return dl_sum, len(dates)