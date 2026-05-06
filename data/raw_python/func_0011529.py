def tell(self):
        """Return the current position in the stream (ignoring bit
        position)

        :returns: int for the position in the stream
        """
        res = self._stream.tell()
        if len(self._bits) > 0:
            res -= 1
        return res