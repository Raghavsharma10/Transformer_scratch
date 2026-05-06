def per_distro_data(self):
        """
        Return download data by distro name and version.

        :return: dict of cache data; keys are datetime objects, values are
          dict of distro name/version (str) to count (int).
        :rtype: dict
        """
        ret = {}
        for cache_date in self.cache_dates:
            data = self._cache_get(cache_date)
            ret[cache_date] = {}
            for distro_name, distro_data in data['by_distro'].items():
                if distro_name.lower() == 'red hat enterprise linux server':
                    distro_name = 'RHEL'
                for distro_ver, count in distro_data.items():
                    ver = self._shorten_version(distro_ver, num_components=1)
                    if distro_name.lower() == 'os x':
                        ver = self._shorten_version(distro_ver,
                                                    num_components=2)
                    k = self._compound_column_value(distro_name, ver)
                    ret[cache_date][k] = count
            if len(ret[cache_date]) == 0:
                ret[cache_date]['unknown'] = 0
        return ret