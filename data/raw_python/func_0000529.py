def to_dict(self):
        """Convert Term into raw dictionary data."""
        return {
            'definition': self.definition,
            'id': self.term_id,
            'image': self.image.to_dict(),
            'rank': self.rank,
            'term': self.term
        }