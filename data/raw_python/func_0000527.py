def to_dict(self):
        """Convert Image into raw dictionary data."""
        if not self.url:
            return None
        return {
            'url': self.url,
            'width': self.width,
            'height': self.height
        }