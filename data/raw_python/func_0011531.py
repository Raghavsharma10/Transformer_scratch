def size(self):
        """Return the size of the stream, or -1 if it cannot
        be determined.
        """
        pos = self._stream.tell()
        # seek to the end of the stream
        self._stream.seek(0,2)
        size = self._stream.tell()
        self._stream.seek(pos, 0)

        return size