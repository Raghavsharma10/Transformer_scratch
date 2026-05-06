def serialize_compound(self, tag):
        """Return the literal representation of a compound tag."""
        separator, fmt = self.comma, '{{{}}}'

        with self.depth():
            if self.should_expand(tag):
                separator, fmt = self.expand(separator, fmt)

            return fmt.format(separator.join(
                f'{self.stringify_compound_key(key)}{self.colon}{self.serialize(value)}'
                for key, value in tag.items()
            ))