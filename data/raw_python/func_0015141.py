def write_string(self, string):
        """Write a string to this packet."""
        self.write_uint16(len(string))
        self.write_bytes(string, len(string))