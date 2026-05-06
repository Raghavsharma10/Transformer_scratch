def per_version_data(self):
        """
        Return download data by version.

        :return: dict of cache data; keys are datetime objects, values are
          dict of version (str) to count (int)
        :rtype: dict
        """
        ret = {}
        for cache_date in self.cache_dates:
            data = self._cache_get(cache_date)
            if len(data['by_version']) == 0:
                data['by_version'] = {'other': 0}
            ret[cache_date] = data['by_version']
        return ret