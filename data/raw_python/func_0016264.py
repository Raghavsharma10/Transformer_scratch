def save(self, **kwargs):
        """Save and return the object (for chaining)."""
        if self.search_terms is None:
            self.search_terms = ""
        super().save(**kwargs)
        return self