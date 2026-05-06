def downloads_per_day(self):
        """
        Return the number of downloads per day, averaged over the past 7 days
        of data.

        :return: average number of downloads per day
        :rtype: int
        """
        count, num_days = self._downloads_for_num_days(7)
        res = ceil(count / num_days)
        logger.debug("Downloads per day = (%d / %d) = %d", count, num_days, res)
        return res