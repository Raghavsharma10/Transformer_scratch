def branchScale(self):
        """See docs for `Model` abstract base class."""
        bs = -(self.prx * scipy.diagonal(self.Prxy, axis1=1, axis2=2)
                ).sum() * self.mu / float(self.nsites)
        assert bs > 0
        return bs