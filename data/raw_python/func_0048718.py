def _inject_format_spec(self, value, format_spec):
        """
        value: '{x}', format_spec: 'f' -> '{x:f}'
        """
        t = type(value)
        return value[:-1] + t(u':') + format_spec + t(u'}')