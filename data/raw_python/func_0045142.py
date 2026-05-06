def finalize(self, sched):
        """Return a reference to the instance of the newly added coroutine."""
        super(AddCoro, self).finalize(sched)
        return self.result