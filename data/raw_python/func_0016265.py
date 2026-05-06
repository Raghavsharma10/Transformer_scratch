def page_slice(self):
        """Return the query from:size tuple (0-based)."""
        return (
            None
            if self.query is None
            else (self.query.get("from", 0), self.query.get("size", 10))
        )