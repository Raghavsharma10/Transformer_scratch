def read_uint16(self):
        """Read 2 bytes."""
        if self.pos + 2 > self.remaining_length:
            return NC.ERR_PROTOCOL
        msb = self.payload[self.pos]
        self.pos += 1
        lsb = self.payload[self.pos]
        self.pos += 1
        
        word = (msb << 8) + lsb
        
        return NC.ERR_SUCCESS, word