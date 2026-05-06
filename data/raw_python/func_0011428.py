def _pfp__parse(self, stream, save_offset=False):
        """Parse the IO stream for this enum

        :stream: An IO stream that can be read from
        :returns: The number of bytes parsed
        """
        res = super(Enum, self)._pfp__parse(stream, save_offset)

        if self._pfp__value in self.enum_vals:
            self.enum_name = self.enum_vals[self._pfp__value]
        else:
            self.enum_name = "?? UNK_ENUM ??"

        return res