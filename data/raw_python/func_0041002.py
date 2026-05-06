def fetch(self, rebuild=False, cache=True):
        """Fetches the table and applies all post processors.
        Args:
            rebuild (bool): Rebuild the table and ignore cache. Default: False
            cache (bool): Cache the finished table for faster future loading.
                Default: True
        """
        if rebuild:
            return self._process_table(cache)
        try:
            return self.read_cache()
        except FileNotFoundError:
            return self._process_table(cache)