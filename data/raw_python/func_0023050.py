def get(self, tags):
        """Find an adequate value for this field from a dict of tags."""
        # Try to find our name
        value = tags.get(self.name, '')

        for name in self.alternate_tags:
            # Iterate of alternates until a non-empty value is found
            value = value or tags.get(name, '')

        # If we still have nothing, return our default
        value = value or self.default
        return value