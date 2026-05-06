def load_all(self, workers=None, limit=None, n_expected=None):
        """Load all instances witih multiple threads.

        :param workers: number of workers to use to load instances, which
                        defaults to what was given in the class initializer
        :param limit: return a maximum, which defaults to no limit

        :param n_expected: rerun the iteration on the data if we didn't find
                           enough data, or more specifically, number of found
                           data points is less than ``n_expected``; defaults to
                           all

        """
        if not self.has_data:
            self._preempt(True)
            # we did the best we could (avoid repeat later in this method)
            n_expected = 0
        keys = tuple(self.delegate.keys())
        if n_expected is not None and len(keys) < n_expected:
            self._preempt(True)
            keys = self.delegate.keys()
        keys = it.islice(limit, keys) if limit is not None else keys
        pool = self._create_thread_pool(workers)
        logger.debug(f'workers={workers}, keys: {keys}')
        try:
            return iter(pool.map(self.delegate.load, keys))
        finally:
            pool.close()