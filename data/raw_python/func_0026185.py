def _decode_element(self, line):
        """Decode element to unicode if necessary."""
        if not self.encoding:
            return line
        if isinstance(line, str) and self.default_encoding:
            return line.decode(self.default_encoding)
        return line