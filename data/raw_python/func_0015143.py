def write_byte(self, byte):
        """Write one byte."""
        self.payload[self.pos] = byte
        self.pos = self.pos + 1