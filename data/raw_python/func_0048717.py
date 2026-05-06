def _inject_conversion(self, value, conversion):
        """
        value: '{x}', conversion: 's' -> '{x!s}'
        """
        t = type(value)
        return value[:-1] + t(u'!') + conversion + t(u'}')