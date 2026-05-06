def serialize_list(self, tag):
        """Return the literal representation of a list tag."""
        separator, fmt = self.comma, '[{}]'

        with self.depth():
            if self.should_expand(tag):
                separator, fmt = self.expand(separator, fmt)

            return fmt.format(separator.join(map(self.serialize, tag)))