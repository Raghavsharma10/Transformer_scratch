def _fix(self, fmt='i'):
        """
        Read pre- or suffix of line at current position with given
        format `fmt` (default 'i').
        """
        fmt = self.endian + fmt
        fix = self.read(struct.calcsize(fmt))
        if fix:
            return struct.unpack(fmt, fix)[0]
        else:
            raise EOFError