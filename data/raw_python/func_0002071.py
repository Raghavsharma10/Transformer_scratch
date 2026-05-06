def _options(self):
        """
        Returns a raw options object

        :rtype: dict
        """
        if self._options_cache is None:
            target_url = self.client.get_url(self._URL_KEY, 'OPTIONS', 'options')
            r = self.client.request('OPTIONS', target_url)
            self._options_cache = r.json()
        return self._options_cache