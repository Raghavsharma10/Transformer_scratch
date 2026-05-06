def readByte(self):
        """
        Reads a byte value from the L{ReadData} stream object.
        
        @rtype: int
        @return: The byte value read from the L{ReadData} stream.
        """
        byte = unpack('B' if not self.signed else 'b', self.readAt(self.offset, 1))[0]
        self.offset += 1
        return byte