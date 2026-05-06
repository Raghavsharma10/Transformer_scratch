def _raw_open(self, flags, mode=0o777):
        """
        Open the file pointed by this path and return a file descriptor,
        as os.open() does.
        """
        return self._accessor.open(self, flags, mode)