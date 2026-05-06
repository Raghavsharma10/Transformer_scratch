def register(self, model, values=None, instance_values=None):
        """
        Registers a model with this group.

        :param values: A list of values that should be incremented \
        whenever invalidate_cache is called for a instance or class \
        of this type.

        :param instance_values: A list of attribute names that will \
        be looked up on the instance of this model that is passed to \
        invalidate_cache. The value resulting from that lookup \
        will then be incremented.
        """

        if model in self._models:
            raise Exception("%s is already registered" % model)

        self._models[model] = CacheConfig(values, instance_values)