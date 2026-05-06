def rev(self, i):
        """Return a clone with a different revision."""
        on = copy(self)
        on.revision = i
        return on