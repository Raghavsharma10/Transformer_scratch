def read_bytes(self, count):
        """Read count number of bytes."""
        if self.pos + count > self.remaining_length:
            return NC.ERR_PROTOCOL, None
        
        ba = bytearray(count)
        for x in xrange(0, count):
            ba[x] = self.payload[self.pos]
            self.pos += 1
        
        return NC.ERR_SUCCESS, ba