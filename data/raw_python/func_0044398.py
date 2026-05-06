def new_tag(self, name: str, category: str=None) -> models.Tag:
        """Create a new tag."""
        new_tag = self.Tag(name=name, category=category)
        return new_tag