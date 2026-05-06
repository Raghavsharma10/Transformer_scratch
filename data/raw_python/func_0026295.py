def writelines(self, lines, fmt):
        """
        Write `lines` with given `format`.
        """
        if isinstance(fmt, basestring):
            fmt = [fmt] * len(lines)
        for f, line in zip(fmt, lines):
            self.writeline(f, line, self.endian)