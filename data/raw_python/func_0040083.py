def fetchref(self, ref):
        """Fetch a particular git ref."""
        log.debug('[%s] Fetching ref: %s', self.name, ref)
        fetch_info = self.repo.remotes.origin.fetch(ref).pop()
        return fetch_info.ref