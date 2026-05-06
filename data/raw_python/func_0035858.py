def _manifest(self):
        """Return manifest content."""
        if self._manifest_cache is None:
            self._manifest_cache = self._storage_broker.get_manifest()

        return self._manifest_cache