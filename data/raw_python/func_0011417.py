def _pfp__parse(self, stream, save_offset=False):
        """Parse the incoming stream

        :stream: Input stream to be parsed
        :returns: Number of bytes parsed

        """
        if save_offset:
            self._pfp__offset = stream.tell()

        res = 0
        for child in self._pfp__children:
            res += child._pfp__parse(stream, save_offset)
        return res