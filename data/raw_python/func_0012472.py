def per_country_data(self):
        """
        Return download data by country.

        :return: dict of cache data; keys are datetime objects, values are
          dict of country (str) to count (int)
        :rtype: dict
        """
        ret = {}
        for cache_date in self.cache_dates:
            data = self._cache_get(cache_date)
            ret[cache_date] = {}
            for cc, count in data['by_country'].items():
                k = '%s (%s)' % (self._alpha2_to_country(cc), cc)
                ret[cache_date][k] = count
            if len(ret[cache_date]) == 0:
                ret[cache_date]['unknown'] = 0
        return ret