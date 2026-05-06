def cache(self):
        """Get the Django cache interface.

        This allows disabling the cache with
        settings.USE_DRF_INSTANCE_CACHE=False.  It also delays import so that
        Django Debug Toolbar will record cache requests.
        """
        if not self._cache:
            use_cache = getattr(settings, 'USE_DRF_INSTANCE_CACHE', True)
            if use_cache:
                from django.core.cache import cache
                self._cache = cache
        return self._cache