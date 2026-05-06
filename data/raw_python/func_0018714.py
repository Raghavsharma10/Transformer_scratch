def write_byte(self, value):
        """Writes a single byte to the packet"""
        format = '!B'
        self.data.append(struct.pack(format, value))
        self.size += 1