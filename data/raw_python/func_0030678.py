def memory_objects_for_hash(self, n):
        """
        Returns a set of :class:`SimMemoryObjects` that contain expressions that contain a variable with the hash
        `h`.
        """
        return set([self[i] for i in self.addrs_for_hash(n)])