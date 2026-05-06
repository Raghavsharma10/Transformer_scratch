def readline(self, fmt=None):
        """
        Return next unformatted "line". If format is given, unpack content,
        otherwise return byte string.
        """
        prefix_size = self._fix()

        if fmt is None:
            content = self.read(prefix_size)
        else:
            fmt = self.endian + fmt
            fmt = _replace_star(fmt, prefix_size)
            content = struct.unpack(fmt, self.read(prefix_size))

        try:
            suffix_size = self._fix()
        except EOFError:
            # when endian is invalid and prefix_size > total file size
            suffix_size = -1

        if prefix_size != suffix_size:
            raise IOError(_FIX_ERROR)

        return content