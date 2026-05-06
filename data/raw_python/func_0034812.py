def put_resource(self, resource):
        """
        Adds a resource back to the pool or discards it if the pool is full.

        :param resource: A resource object.

        :raises UnknownResourceError: If resource was not made by the
                                        pool.
        """
        rtracker = self._get_tracker(resource)

        try:
            self._put(rtracker)
        except PoolFullError:
            self._remove(rtracker)