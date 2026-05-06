def _pfp__parse(self, stream, save_offset=False):
        """Read from the stream until the string is null-terminated

        :stream: The input stream
        :returns: None

        """
        if save_offset:
            self._pfp__offset = stream.tell()

        res = utils.binary("")
        while True:
            byte = utils.binary(stream.read(self.read_size))
            if len(byte) < self.read_size:
                raise errors.PrematureEOF()
            # note that the null terminator must be added back when
            # built again!
            if byte == self.terminator:
                break
            res += byte
        self._pfp__value = res