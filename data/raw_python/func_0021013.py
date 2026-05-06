def scan_items(self):
        """
        Yield each of the ``(key, value)`` pairs from the collection, without
        pulling them all into memory.

        .. warning::
            This method is not available on the dictionary collections provided
            by Python.

            This method may return the same (key, value) pair multiple times.
            See the `Redis SCAN documentation
            <http://redis.io/commands/scan#scan-guarantees>`_ for details.
        """
        for k, v in self.redis.hscan_iter(self.key):
            yield self._unpickle_key(k), self._unpickle(v)