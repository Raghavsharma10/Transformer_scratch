def _pfp__parse(self, stream, save_offset=False):
        """Parse the IO stream for this numeric field

        :stream: An IO stream that can be read from
        :returns: The number of bytes parsed
        """
        if save_offset:
            self._pfp__offset = stream.tell()

        if self.bitsize is None:
            raw_data = stream.read(self.width)
            data = utils.binary(raw_data)

        else:
            bits = self.bitfield_rw.read_bits(stream, self.bitsize, self.bitfield_padded, self.bitfield_left_right, self.endian)
            width_diff = self.width  - (len(bits)//8) - 1
            bits_diff = 8 - (len(bits) % 8)

            padding = [0] * (width_diff * 8 + bits_diff)

            bits = padding + bits

            data = bitwrap.bits_to_bytes(bits)
            if self.endian == LITTLE_ENDIAN:
                # reverse the data
                data = data[::-1]

        if len(data) < self.width:
            raise errors.PrematureEOF()

        self._pfp__data = data
        self._pfp__value = struct.unpack(
            "{}{}".format(self.endian, self.format),
            data
        )[0]

        return self.width