def raw_get(self, *raw_args, **raw_kwargs):
        """
        Retrieve the item (tuple of value and expiry) that is actually in the cache,
        without causing a refresh.
        """

        args = self.prepare_args(*raw_args)
        kwargs = self.prepare_kwargs(**raw_kwargs)

        key = self.key(*args, **kwargs)

        return self.cache.get(key)