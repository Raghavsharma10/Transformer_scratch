def branchScale(self):
        """See docs for `Model` abstract base class."""
        bscales = [m.branchScale for m in self._models]
        return (self.catweights * bscales).sum()