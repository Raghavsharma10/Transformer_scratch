def dispatch(self, request, *args, **kwargs):
        """
        Overrides Django's default dispatch to provide caching.

        If the should_cache method returns True, this will call
        two functions get_cache_version and get_cache_prefix
        the results of those two functions are combined and passed to
        the standard django caching middleware.
        """

        self.request = request
        self.args = args
        self.kwargs = kwargs
        self.cache_middleware = None
        response = None

        if self.should_cache():
            prefix = "%s:%s" % (self.get_cache_version(),
                                self.get_cache_prefix())

            # Using middleware here since that is what the decorator uses
            # internally and it avoids making this code all complicated with
            # all sorts of wrappers.
            self.set_cache_middleware(self.cache_time, prefix)
            response = self.cache_middleware.process_request(self.request)
        else:
            self.set_do_not_cache()

        if not response:
            response = super(CacheView, self).dispatch(self.request, *args,
                                                       **kwargs)

        return self._finalize_cached_response(request, response)