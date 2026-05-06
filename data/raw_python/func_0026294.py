def writeline(self, fmt, *args):
        """
        Write `line` (list of objects) with given `fmt` to file. The
        `line` will be chained if object is iterable (except for
        basestrings).
        """
        fmt = self.endian + fmt
        size = struct.calcsize(fmt)

        fix = struct.pack(self.endian + 'i', size)
        line = struct.pack(fmt, *args)

        self.write(fix)
        self.write(line)
        self.write(fix)