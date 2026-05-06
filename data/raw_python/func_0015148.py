def read_string(self):
        """Read string."""
        rc, length = self.read_uint16()
        
        if rc != NC.ERR_SUCCESS:
            return rc, None
        
        if self.pos + length > self.remaining_length:
            return NC.ERR_PROTOCOL, None
        
        ba = bytearray(length)
        if ba is None:
            return NC.ERR_NO_MEM, None
        
        for x in xrange(0, length):
            ba[x] = self.payload[self.pos]
            self.pos += 1
        
        return NC.ERR_SUCCESS, ba