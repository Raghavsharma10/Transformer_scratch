def seek(self, pos, seek_type=0):
        """Seek to the specified position in the stream with seek_type.
        Unflushed bits will be discarded in the case of a seek.

        The stream will also keep track of which bytes have and have
        not been consumed so that the dom will capture all of the
        bytes in the stream.

        :pos: offset
        :seek_type: direction
        :returns: TODO

        """
        self._bits.clear()
        return self._stream.seek(pos, seek_type)