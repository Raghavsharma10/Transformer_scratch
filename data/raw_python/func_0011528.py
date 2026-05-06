def write_bits(self, bits):
        """Write the bits to the stream.

        Add the bits to the existing unflushed bits and write
        complete bytes to the stream.
        """
        for bit in bits:
            self._bits.append(bit)

        while len(self._bits) >= 8:
            byte_bits = [self._bits.popleft() for x in six.moves.range(8)]
            byte = bits_to_bytes(byte_bits)
            self._stream.write(byte)