def write_int(self, value):
        """Writes an unsigned integer to the packet"""
        format = '!I'
        self.data.append(struct.pack(format, int(value)))
        self.size += 4