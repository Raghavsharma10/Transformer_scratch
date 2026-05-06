def get_size_in_bytes(self, handle):
        """Return the size in bytes."""
        fpath = self._fpath_from_handle(handle)
        return os.stat(fpath).st_size