def to_representation(self, value):
        """Convert to natural key."""
        content_type = ContentType.objects.get_for_id(value)
        return "_".join(content_type.natural_key())