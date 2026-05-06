def create_cache(self, **kwargs):
        """
        Creates an instance of the Cache Service.
        """
        cache = predix.admin.cache.Cache(**kwargs)
        cache.create(**kwargs)
        cache.add_to_manifest(self)
        return cache