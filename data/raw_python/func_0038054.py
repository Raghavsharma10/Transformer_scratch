def to_internal_value(self, value):
        """Convert to integer id."""
        natural_key = value.split("_")
        content_type = ContentType.objects.get_by_natural_key(*natural_key)
        return content_type.id