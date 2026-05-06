def _escape(self, bits):
        """
        value: 'foobar {' -> 'foobar {{'
        value: 'x}' -> 'x}}'
        """
        # for value, field_name, format_spec, conversion in bits:
        while True:
            try:
                value, field_name, format_spec, conversion = next(bits)
                if value:
                    end = value[-1]
                    if end in (u'{', u'}'):
                        value += end
                yield value, field_name, format_spec, conversion
            except StopIteration:
                break