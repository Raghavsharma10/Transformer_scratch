def serialize_array(self, tag):
        """Return the literal representation of an array tag."""
        elements = self.comma.join(f'{el}{tag.item_suffix}' for el in tag)
        return f'[{tag.array_prefix}{self.semicolon}{elements}]'