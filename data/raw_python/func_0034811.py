def get_resource(self, resource_wrapper=None):
        """
        Returns a ``Resource`` instance.

        :param resource_wrapper: A Resource subclass.
        :return: A ``Resource`` instance.

        :raises PoolEmptyError: If attempt to get resource fails or times
            out.
        """
        rtracker = None

        if resource_wrapper is None:
            resource_wrapper = self._resource_wrapper

        if self.empty():
            self._harvest_lost_resources()

        try:
            rtracker = self._get(0)
        except PoolEmptyError:
            pass

        if rtracker is None:
            # Could not find resource, try to make one.
            try:
                rtracker = self._make_resource()
            except PoolFullError:
                pass

        if rtracker is None:
            # Could not find or make resource, so must wait for a resource
            # to be returned to the pool.
            try:
                rtracker = self._get(timeout=self._timeout)
            except PoolEmptyError:
                pass

        if rtracker is None:
            raise PoolEmptyError

        # Ensure resource is active.
        if not self.ping(rtracker.resource):
            # Lock here to prevent another thread creating a resource in the
            # index that will have this resource removed. This ensures there
            # will be space for _make_resource() to place a newly created
            # resource.
            with self._lock:
                self._remove(rtracker)
                rtracker = self._make_resource()

        # Ensure all resources leave pool with same attributes.
        # normalize_connection() is used since it calls
        # normalize_resource(), so if a user implements either one, the
        # resource will still be normalized. This will be changed in 1.0 to
        # call normalize_resource() when normalize_connection() is
        # removed.
        self.normalize_connection(rtracker.resource)

        return rtracker.wrap_resource(self, resource_wrapper)