def write_bits(self, stream, raw_bits, padded, left_right, endian):
        """Write the bits. Once the size of the written bits is equal
        to the number of the reserved bits, flush it to the stream
        """
        if padded:
            if left_right:
                self._write_bits += raw_bits
            else:
                self._write_bits = raw_bits + self._write_bits

            if len(self._write_bits) == self.reserved_bits:
                bits = self._endian_transform(self._write_bits, endian)

                # if it's padded, and all of the bits in the field weren't used,
                # we need to flush out the unused bits
                # TODO should we save the value of the unused bits so the data that
                # is written out matches exactly what was read?
                if self.reserved_bits < self.cls.width * 8:
                    filler = [0] * ((self.cls.width * 8) - self.reserved_bits)
                    if left_right:
                        bits += filler
                    else:
                        bits = filler + bits

                stream.write_bits(bits)
                self._write_bits = []

        else:
            # if an unpadded field ended up using the same BitfieldRW and
            # as a previous padded field, there will be unwritten bits left in
            # self._write_bits. These need to be flushed out as well
            if len(self._write_bits) > 0:
                stream.write_bits(self._write_bits)
                self._write_bits = []

            stream.write_bits(raw_bits)