def register_model(self, key, *models, **kwargs):
        """
        Register a cache_group with this manager.

        Use this method to register more simple
        groups where all models share the same parameters.

        Any arguments are treated as models that you would like
        to register.

        Any keyword arguments received are passed to the
        register method when registering each model.

        :param key: The key to register this group as. \
        Raises an exception if the key is already registered.
        """

        cache_group = CacheGroup(key)
        for model in models:
            cache_group.register(model, **kwargs)

        self.register_cache(cache_group)