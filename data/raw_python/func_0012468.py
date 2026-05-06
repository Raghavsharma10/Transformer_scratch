def per_file_type_data(self):
        """
        Return download data by file type.

        :return: dict of cache data; keys are datetime objects, values are
          dict of file type (str) to count (int)
        :rtype: dict
        """
        ret = {}
        for cache_date in self.cache_dates:
            data = self._cache_get(cache_date)
            if len(data['by_file_type']) == 0:
                data['by_file_type'] = {'other': 0}
            ret[cache_date] = data['by_file_type']
        return ret