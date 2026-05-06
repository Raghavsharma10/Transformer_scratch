def invalidate_cache(self, klass, extra=None, **kwargs):
        """
        Invalidate a cache for a specific class.

        This will loop through all registered groups that have registered
        the given model class and call their invalidate_cache method.

        All keyword arguments will be directly passed through to the
        group's invalidate_cache method, with the exception of **extra**
        as noted below.

        :param klass: The model class that need some invalidation.

        :param extra: A dictionary where the key corresponds to the name \
        of a group where this model is registered and a value that is a \
        list that will be passed as the extra keyword argument when \
        calling invalidate_cache on that group. In this way you can \
        specify specific extra values to invalidate only for specific \
        groups.
        """

        extra = extra or kwargs.pop('extra', {})
        for group in self._registry.values():
            if klass in group.models:
                e = extra.get(group.key)
                group.invalidate_cache(klass, extra=e, **kwargs)