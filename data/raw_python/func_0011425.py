def _pfp__build(self, stream=None, save_offset=False):
        """Build the field and write the result into the stream

        :stream: An IO stream that can be written to
        :returns: None

        """
        if stream is not None and save_offset:
            self._pfp__offset = stream.tell()

        if self.bitsize is None:
            data = struct.pack(
                "{}{}".format(self.endian, self.format),
                self._pfp__value
            )
            if stream is not None:
                stream.write(data)
                return len(data)
            else:
                return data
        else:
            data = struct.pack(
                "{}{}".format(BIG_ENDIAN, self.format),
                self._pfp__value
            )

            num_bytes = int(math.ceil(self.bitsize / 8.0))
            bit_data = data[-num_bytes:]

            raw_bits = bitwrap.bytes_to_bits(bit_data)
            bits = raw_bits[-self.bitsize:]

            if stream is not None:
                self.bitfield_rw.write_bits(stream, bits, self.bitfield_padded, self.bitfield_left_right, self.endian)
                return len(bits) // 8
            else:
                # TODO this can't be right....
                return bits