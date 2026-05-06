def bundle(self, name: str) -> models.Bundle:
        """Fetch a bundle from the store."""
        return self.Bundle.filter_by(name=name).first()