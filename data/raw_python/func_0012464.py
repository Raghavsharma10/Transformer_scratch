def _cache_get(self, date):
        """
        Return cache data for the specified day; cache locally in this class.

        :param date: date to get data for
        :type date: datetime.datetime
        :return: cache data for date
        :rtype: dict
        """
        if date in self.cache_data:
            logger.debug('Using class-cached data for date %s',
                         date.strftime('%Y-%m-%d'))
            return self.cache_data[date]
        logger.debug('Getting data from cache for date %s',
                     date.strftime('%Y-%m-%d'))
        data = self.cache.get(self.project_name, date)
        self.cache_data[date] = data
        return data