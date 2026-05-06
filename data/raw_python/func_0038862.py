def _value_data(self, value):
        """Parses binary and unidentified values."""
        return codecs.decode(
            codecs.encode(self.value_value(value)[1], 'base64'), 'utf8')