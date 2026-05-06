def close(self):
        """Close the stream
        """
        self.closed = True
        self._flush_bits_to_stream()
        self._stream.close()