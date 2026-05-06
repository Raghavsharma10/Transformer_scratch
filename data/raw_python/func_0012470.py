def per_implementation_data(self):
        """
        Return download data by python impelementation name and version.

        :return: dict of cache data; keys are datetime objects, values are
          dict of implementation name/version (str) to count (int).
        :rtype: dict
        """
        ret = {}
        for cache_date in self.cache_dates:
            data = self._cache_get(cache_date)
            ret[cache_date] = {}
            for impl_name, impl_data in data['by_implementation'].items():
                for impl_ver, count in impl_data.items():
                    k = self._compound_column_value(
                        impl_name,
                        self._shorten_version(impl_ver)
                    )
                    ret[cache_date][k] = count
            if len(ret[cache_date]) == 0:
                ret[cache_date]['unknown'] = 0
        return ret