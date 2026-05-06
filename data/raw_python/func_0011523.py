def is_eof(self):
        """Return if the stream has reached EOF or not
        without discarding any unflushed bits

        :returns: True/False
        """
        pos = self._stream.tell()
        byte = self._stream.read(1)
        self._stream.seek(pos, 0)

        return utils.binary(byte) == utils.binary("")