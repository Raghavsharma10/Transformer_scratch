def get_unset_cache(self):
        """return : returns a tuple (num_of_not_None_caches, [list of unset caches endpoint])
        """
        caches = []
        if self._cached_api_global_response is None:
            caches.append('global')
        if self._cached_api_ticker_response is None:
            caches.append('ticker')
        return (len(caches), caches)