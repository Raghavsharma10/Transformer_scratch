def write_bytes(self, data, n):
        """Write n number of bytes to this packet."""
        for pos in xrange(0, n):
            self.payload[self.pos + pos] = data[pos]
            
        self.pos += n