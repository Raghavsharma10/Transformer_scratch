def read(self, n=-1):
        """Read and return up to n bytes.
        If the argument is omitted, None, or negative, data is read and returned until EOF is reached..
        """

        buf = b''
        while n < 0 or n is None or n > len(buf):
            data = self.read1(n)
            if len(data) == 0:
                return buf

            buf += data

        return buf