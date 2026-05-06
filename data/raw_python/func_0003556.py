def _init(self):
        """Read the b"\\r\\n" at the end of the message."""
        read_values = []
        read = self._file.read
        last = read(1)
        current = read(1)
        while last != b'' and current != b'' and not \
                (last == b'\r' and current == b'\n'):
            read_values.append(last)
            last = current
            current = read(1)
        if current == b'' and last != b'\r':
            read_values.append(last)
        self._bytes = b''.join(read_values)