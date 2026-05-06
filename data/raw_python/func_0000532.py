def to_dict(self):
        """Convert WordSet into raw dictionary data."""
        return {
            'id': self.set_id,
            'title': self.title,
            'terms': [term.to_dict() for term in self.terms]
        }