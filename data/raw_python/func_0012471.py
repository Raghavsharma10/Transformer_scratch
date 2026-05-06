def per_system_data(self):
        """
        Return download data by system.

        :return: dict of cache data; keys are datetime objects, values are
          dict of system (str) to count (int)
        :rtype: dict
        """
        ret = {}
        for cache_date in self.cache_dates:
            data = self._cache_get(cache_date)
            ret[cache_date] = {
                self._column_value(x): data['by_system'][x]
                for x in data['by_system']
            }
            if len(ret[cache_date]) == 0:
                ret[cache_date]['unknown'] = 0
        return ret