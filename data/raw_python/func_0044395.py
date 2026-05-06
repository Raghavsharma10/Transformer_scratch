def new_bundle(self, name: str, created_at: dt.datetime=None) -> models.Bundle:
        """Create a new file bundle."""
        new_bundle = self.Bundle(name=name, created_at=created_at)
        return new_bundle