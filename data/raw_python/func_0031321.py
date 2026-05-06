def clean_new(self, value):
        """Return a new object instantiated with cleaned data."""
        value = self.schema_class(value).full_clean()
        return self.object_class(**value)