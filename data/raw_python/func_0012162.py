def select(self, fixed):
        """
        Return a subset of variables according to ``fixed``.
        """
        names = [n for n in self.names() if self[n].isfixed == fixed]
        return Variables({n: self[n] for n in names})