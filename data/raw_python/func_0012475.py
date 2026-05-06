def downloads_per_week(self):
        """
        Return the number of downloads in the last 7 days.

        :return: number of downloads in the last 7 days; if we have less than
          7 days of data, returns None.
        :rtype: int
        """
        if len(self.cache_dates) < 7:
            logger.error("Only have %d days of data; cannot calculate "
                         "downloads per week", len(self.cache_dates))
            return None
        count, _ = self._downloads_for_num_days(7)
        logger.debug("Downloads per week = %d", count)
        return count