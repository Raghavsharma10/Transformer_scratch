def close(self):
        """
        Returns the resource to the resource pool.
        """
        if self._resource is not None:
            self._pool.put_resource(self._resource)
            self._resource = None
            self._pool = None