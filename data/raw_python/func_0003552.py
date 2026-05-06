def read_end_of_message(self):
        """Read the b"\\r\\n" at the end of the message."""
        read = self._file.read
        last = read(1)
        current = read(1)
        while last != b'' and current != b'' and not \
                (last == b'\r' and current == b'\n'):
            last = current
            current = read(1)