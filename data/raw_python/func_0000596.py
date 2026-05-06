def parse_value(self, value):
        """Parse value to proper model type."""
        if not isinstance(value, dict):
            return value

        embed_type = self._get_embed_type()
        return embed_type(**value)