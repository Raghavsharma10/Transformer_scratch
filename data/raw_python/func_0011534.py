def _flush_bits_to_stream(self):
        """Flush the bits to the stream. This is used when
        a few bits have been read and ``self._bits`` contains unconsumed/
        flushed bits when data is to be written to the stream
        """
        if len(self._bits) == 0:
            return 0

        bits = list(self._bits)

        diff = 8 - (len(bits) % 8)
        padding = [0] * diff

        bits = bits + padding

        self._stream.write(bits_to_bytes(bits))

        self._bits.clear()