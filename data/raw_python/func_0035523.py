def branchScale(self):
        """See docs for `Model` abstract base class."""
        bs = -(self.Phi_x * scipy.diagonal(self.Pxy[0])).sum() * self.mu
        assert bs > 0
        return bs