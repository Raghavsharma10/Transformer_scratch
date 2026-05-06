def _have_cache_for_date(self, dt):
        """
        Return True if we have cached data for all projects for the specified
        datetime. Return False otherwise.

        :param dt: datetime to find cache for
        :type dt: datetime.datetime
        :return: True if we have cache for all projects for this date, False
          otherwise
        :rtype: bool
        """
        for p in self.projects:
            if self.cache.get(p, dt) is None:
                return False
        return True