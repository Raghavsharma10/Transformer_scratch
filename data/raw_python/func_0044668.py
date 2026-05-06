def read(self, num_bytes):
        """Read `num_bytes` from the compressed data chunks.

        Data is returned as `bytes` of length `num_bytes`

        Will raise an EOFError if data is unavailable.

        Note: Will always return `num_bytes` of data (unlike the file read method).

        """
        while len(self.decoded) < num_bytes:
            try:
                tag, data = next(self.chunks)
            except StopIteration:
                raise EOFError()
            if tag != b'IDAT':
                continue
            self.decoded += self.decompressor.decompress(data)

        r = self.decoded[:num_bytes]
        self.decoded = self.decoded[num_bytes:]
        return r