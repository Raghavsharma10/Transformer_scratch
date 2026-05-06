def scan_items(self):
        """
        Yield each of the ``(member, score)`` tuples from the collection,
        without pulling them all into memory.

        .. warning::
            This method may return the same (member, score) tuple multiple
            times.
            See the `Redis SCAN documentation
            <http://redis.io/commands/scan#scan-guarantees>`_ for details.
        """
        for m, s in self.redis.zscan_iter(self.key):
            yield self._unpickle(m), s