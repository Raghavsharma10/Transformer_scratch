def per_installer_data(self):
        """
        Return download data by installer name and version.

        :return: dict of cache data; keys are datetime objects, values are
          dict of installer name/version (str) to count (int).
        :rtype: dict
        """
        ret = {}
        for cache_date in self.cache_dates:
            data = self._cache_get(cache_date)
            ret[cache_date] = {}
            for inst_name, inst_data in data['by_installer'].items():
                for inst_ver, count in inst_data.items():
                    k = self._compound_column_value(
                        inst_name,
                        self._shorten_version(inst_ver)
                    )
                    ret[cache_date][k] = count
            if len(ret[cache_date]) == 0:
                ret[cache_date]['unknown'] = 0
        return ret