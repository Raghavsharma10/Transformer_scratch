def get_cache_prefix(self, prefix=''):
        """
        Hook for any extra data you would like
        to prepend to your cache key.

        The default implementation ensures that ajax not non
        ajax requests are cached separately. This can easily
        be extended to differentiate on other criteria
        like mobile os' for example.
        """

        if settings.CACHE_MIDDLEWARE_KEY_PREFIX:
            prefix += settings.CACHE_MIDDLEWARE_KEY_PREFIX

        if self.request.is_ajax():
            prefix += 'ajax'

        return prefix