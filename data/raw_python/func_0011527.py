def write(self, data):
        """Write data to the stream

        :data: the data to write to the stream
        :returns: None
        """
        if self.padded:
            # flush out any remaining bits first
            if len(self._bits) > 0:
                self._flush_bits_to_stream()
            self._stream.write(data)
        else:
            # nothing to do here
            if len(data) == 0:
                return

            bits = bytes_to_bits(data)
            self.write_bits(bits)