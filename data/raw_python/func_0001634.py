def load_stats(self, cache=None, wait=None):
        """ Load and cache the webpack-stats file """
        if cache is None:
            cache = not self.debug
        if wait is None:
            wait = self.debug
        if not cache or self._stats is None:
            self._stats = self._load_stats()
            start = time.time()
            while wait and self._stats.get('status') == 'compiling':
                if self.timeout and (time.time() - start > self.timeout):
                    raise RuntimeError("Webpack {0!r} timed out while compiling"
                                       .format(self.stats_file.path))
                time.sleep(0.1)
                self._stats = self._load_stats()
        return self._stats