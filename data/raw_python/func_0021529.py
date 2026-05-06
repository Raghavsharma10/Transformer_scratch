def readAlignedString(self, align = 4):
        """ 
        Reads an ASCII string aligned to the next align-bytes boundary.
        
        @type align: int
        @param align: (Optional) The value we want the ASCII string to be aligned.
        
        @rtype: str
        @return: A 4-bytes aligned (default) ASCII string.
        """
        s = self.readString()
        r = align - len(s) % align
        while r:
            s += self.data[self.offset]
            self.offset += 1
            r -= 1
        return s.rstrip("\x00")