def create(self, **fields):
        """Create new entry."""
        entry = self.instance(**fields)
        entry.save()
        return entry