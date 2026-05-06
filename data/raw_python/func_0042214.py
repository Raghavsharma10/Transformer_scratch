def get_as_string(self, request, *args, **kwargs):
        """
        Should only be used when inheriting from cms View.

        Gets the response as a string and caches it with a
        separate prefix
        """

        value = None
        cache = None
        prefix = None
        if self.should_cache():
            prefix = "%s:%s:string" % (self.get_cache_version(),
                                self.get_cache_prefix())
            cache = router.router.get_cache(prefix)
            value = cache.get(prefix)

        if not value:
            value = super(CacheView, self).get_as_string(request, *args,
                                                         **kwargs)
            if self.should_cache() and value and \
                    getattr(self.request, '_cache_update_cache', False):
                cache.set(prefix, value, self.cache_time)

        return value