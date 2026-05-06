def get_hash(self, handle):
        """Return the hash."""
        fpath = self._fpath_from_handle(handle)
        return DiskStorageBroker.hasher(fpath)