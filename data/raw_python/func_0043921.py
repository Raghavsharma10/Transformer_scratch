def release_filter(self, value):
        """Validate the release filter."""
        compiled_pattern = coerce_pattern(value)
        if compiled_pattern.groups > 1:
            raise ValueError(compact("""
                Release filter regular expression pattern is expected to have
                zero or one capture group, but it has {count} instead!
            """, count=compiled_pattern.groups))
        set_property(self, 'release_filter', value)
        set_property(self, 'compiled_filter', compiled_pattern)