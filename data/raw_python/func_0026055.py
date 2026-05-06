def storeFromWav(self, uploadCacheEntry, start, end):
        """
        Stores a new item in the cache.
        :param name: file name.
        :param start: start time.
        :param end: end time.
        :return: true if stored.
        """
        prefix = uploadCacheEntry['name'] + '_' + start + '_' + end
        match = next((x for x in self._cache.values() if x['type'] == 'wav' and x['name'].startswith(prefix)), None)
        if match is None:
            cached = [
                {
                    'name': prefix + '_' + n,
                    'analysis': n,
                    'start': start,
                    'end': end,
                    'type': 'wav',
                    'filename': uploadCacheEntry['name']
                } for n in ['spectrum', 'peakSpectrum']
            ]
            for cache in cached:
                self._cache[cache['name']] = cache
            self.writeCache()
            return True
        return False