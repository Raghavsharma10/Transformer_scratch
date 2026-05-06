def parse_string(self):
        """Parse a regular unquoted string from the token stream."""
        aliased_value = LITERAL_ALIASES.get(self.current_token.value.lower())
        if aliased_value is not None:
            return aliased_value
        return String(self.current_token.value)