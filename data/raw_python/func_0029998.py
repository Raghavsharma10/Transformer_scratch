def is_finalized(self):
        """Return True if the bundle is installed."""

        return self.state == self.STATES.FINALIZED or self.state == self.STATES.INSTALLED