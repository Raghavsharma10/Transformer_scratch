def stringify_compound_key(self, key):
        """Escape the compound key if it can't be represented unquoted."""
        if UNQUOTED_COMPOUND_KEY.match(key):
            return key
        return self.escape_string(key)