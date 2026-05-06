def read_byte(self):
        """Read a byte."""
        if self.pos + 1 > self.remaining_length:
            return NC.ERR_PROTOCOL, None
        
        byte = self.payload[self.pos]
        self.pos += 1
        
        return NC.ERR_SUCCESS, byte