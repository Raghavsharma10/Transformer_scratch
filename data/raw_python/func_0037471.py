def data(self):
        """Fetch latest data from PyPI, and cache for 30s."""
        key = cache_key(self.name)
        data = cache.get(key)
        if data is None:
            logger.debug("Updating package info for %s from PyPI.", self.name)
            data = requests.get(self.url).json()
            cache.set(key, data, PYPI_CACHE_EXPIRY)
        return data