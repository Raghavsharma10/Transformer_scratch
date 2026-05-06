def readDword(self):
        """
        Reads a dword value from the L{ReadData} stream object.
        
        @rtype: int
        @return: The dword value read from the L{ReadData} stream.
        """
        dword = unpack(self.endianness + ('L' if not self.signed else 'l'), self.readAt(self.offset,  4))[0]
        self.offset += 4
        return dword