def tag(self, name: str) -> models.Tag:
        """Fetch a tag from the database."""
        return self.Tag.filter_by(name=name).first()