def wrap_resource(self, pool, resource_wrapper):
        """
        Return a resource wrapped in ``resource_wrapper``.

        :param pool: A pool instance.
        :type pool: :class:`CuttlePool`
        :param resource_wrapper: A wrapper class for the resource.
        :type resource_wrapper: :class:`Resource`
        :return: A wrapped resource.
        :rtype: :class:`Resource`
        """
        resource = resource_wrapper(self.resource, pool)
        self._weakref = weakref.ref(resource)
        return resource