def register_cache(self, cache_group):
        """
        Register a cache_group with this manager.

        Use this method to register more complicated
        groups that you create yourself. Such as if you
        need to register several models each with different
        parameters.

        :param cache_group: The group to register. \
        The group is registered with the cache_group key attribute. \
        Raises an exception if the key is already registered.
        """

        if cache_group.key in self._registry:
            raise Exception("%s is already registered" % cache_group.key)
        self._registry[cache_group.key] = cache_group