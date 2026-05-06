def read(self, num):
        """Read ``num`` number of bytes from the stream. Note that this will
        automatically resets/ends the current bit-reading if it does not
        end on an even byte AND ``self.padded`` is True. If ``self.padded`` is
        True, then the entire stream is treated as a bitstream.

        :num: number of bytes to read
        :returns: the read bytes, or empty string if EOF has been reached
        """
        start_pos = self.tell()

        if self.padded:
            # we toss out any uneven bytes
            self._bits.clear()
            res = utils.binary(self._stream.read(num))
        else:
            bits = self.read_bits(num * 8)
            res = bits_to_bytes(bits)
            res = utils.binary(res)

        end_pos = self.tell()
        self._update_consumed_ranges(start_pos, end_pos)

        return res