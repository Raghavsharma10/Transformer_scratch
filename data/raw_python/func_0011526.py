def read_bits(self, num):
        """Read ``num`` number of bits from the stream

        :num: number of bits to read
        :returns: a list of ``num`` bits, or an empty list if EOF has been reached
        """
        if num > len(self._bits):
            needed = num - len(self._bits)
            num_bytes = int(math.ceil(needed / 8.0))
            read_bytes = self._stream.read(num_bytes)

            for bit in bytes_to_bits(read_bytes):
                self._bits.append(bit)

        res = []
        while len(res) < num and len(self._bits) > 0:
            res.append(self._bits.popleft())

        return res