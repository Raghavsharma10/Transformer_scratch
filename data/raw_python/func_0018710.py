def read_utf(self, offset, len):
        """Reads a UTF-8 string of a given length from the packet"""
        try:
            result = self.data[offset:offset + len].decode('utf-8')
        except UnicodeDecodeError:
            result = str('')
        return result