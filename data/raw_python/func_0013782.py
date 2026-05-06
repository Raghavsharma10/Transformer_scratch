def _parse_bool(self, value):
        """Helper function to parse default boolean values."""
        if isinstance(value, string_types):
            return value.strip().lower() in ['true', '1', 't', 'y', 'yes']
        return bool(value)