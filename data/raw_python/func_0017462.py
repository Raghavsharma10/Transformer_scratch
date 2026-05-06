def dump(self, format=None, **kwargs):
        """Dump the timeseries using a specific ``format``.
        """
        formatter = Formatters.get(format, None)
        if not format:
            return self.display()
        elif not formatter:
            raise FormattingException('Formatter %s not available' % format)
        else:
            return formatter(self, **kwargs)