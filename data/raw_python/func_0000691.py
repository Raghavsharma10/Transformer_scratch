def set(self, *raw_args, **raw_kwargs):
        """
        Manually set the cache value with its appropriate expiry.
        """
        if self.set_data_kwarg in raw_kwargs:
            data = raw_kwargs.pop(self.set_data_kwarg)
        else:
            raw_args = list(raw_args)
            data = raw_args.pop()

        args = self.prepare_args(*raw_args)
        kwargs = self.prepare_kwargs(**raw_kwargs)

        key = self.key(*args, **kwargs)

        expiry = self.expiry(*args, **kwargs)

        logger.debug("Setting %s cache with key '%s', args '%r', kwargs '%r', expiry '%r'",
                     self.class_path, key, args, kwargs, expiry)

        self.store(key, expiry, data)