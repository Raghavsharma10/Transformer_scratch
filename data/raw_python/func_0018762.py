def _persist(self) -> None:
        """
        Persists the current data group
        """
        if self._store:
            self._store.save(self._key, self._snapshot)