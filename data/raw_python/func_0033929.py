def modify(self, *, sort=None, purge=False, done=None):
        """Calls Model._modifyInternal after loading the database."""
        return self._modifyInternal(sort=sort, purge=purge, done=done)