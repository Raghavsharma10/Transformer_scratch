def invalidate_cache(self, klass, instance=None, extra=None,
                         force_all=False):
        """
        Use this method to invalidate keys related to a particular
        model or instance. Invalidating a cache is really just
        incrementing the version for the right key(s).

        :param klass: The model class you are invalidating. If the given \
        class was not registered with this group no action will be taken.

        :param instance: The instance you want to use with the registered\
        instance_values. Usually the instance that was just saved. \
        Defaults to None.

        :param extra: A list of extra values that you would like incremented \
        in addition to what was registered for this model.

        :param force_all: Ignore all registered values and provided \
        arguments and increment the major version for this group.
        """

        values = self._get_cache_extras(klass, instance=instance,
                                        extra=extra, force_all=force_all)

        if values == CacheConfig.ALL:
            self._increment_version()
        elif values:
            for value in values:
                self._increment_version(extra=value)