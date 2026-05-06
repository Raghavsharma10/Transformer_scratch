def modifyInPlace(self, *, sort=None, purge=False, done=None):
        """Like Model.modify, but changes existing database instead of
        returning a new one."""
        self.data = self.modify(sort=sort, purge=purge, done=done)