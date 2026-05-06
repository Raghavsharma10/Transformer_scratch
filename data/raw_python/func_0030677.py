def memory_objects_for_name(self, n):
        """
        Returns a set of :class:`SimMemoryObjects` that contain expressions that contain a variable with the name of
        `n`.

        This is useful for replacing those values in one fell swoop with :func:`replace_memory_object()`, even if
        they have been partially overwritten.
        """
        return set([self[i] for i in self.addrs_for_name(n)])