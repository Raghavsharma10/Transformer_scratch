def _updateRepo(self, func, *args, **kwargs):
        """
        Runs the specified function that updates the repo with the specified
        arguments. This method ensures that all updates are transactional,
        so that if any part of the update fails no changes are made to the
        repo.
        """
        # TODO how do we make this properly transactional?
        self._repo.open(datarepo.MODE_WRITE)
        try:
            func(*args, **kwargs)
            self._repo.commit()
        finally:
            self._repo.close()