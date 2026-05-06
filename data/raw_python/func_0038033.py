def put_cache_results(self, key, func_akw, set_cache_cb):
        """Put function results into cache."""
        args, kwargs = func_akw

        # get function results
        func_results = self.func(*args, **kwargs)

        # optionally add results to cache
        if set_cache_cb(func_results):
            self[key] = func_results
        return func_results