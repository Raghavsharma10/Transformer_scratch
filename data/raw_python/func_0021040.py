def scan_elements(self):
        """
        Yield each of the elements from the collection, without pulling them
        all into memory.

        .. warning::
            This method is not available on the set collections provided
            by Python.

            This method may return the element multiple times.
            See the `Redis SCAN documentation
            <http://redis.io/commands/scan#scan-guarantees>`_ for details.
        """
        for x in self.redis.sscan_iter(self.key):
            yield self._unpickle(x)