def readWord(self):
        """
        Reads a word value from the L{ReadData} stream object.
        
        @rtype: int
        @return: The word value read from the L{ReadData} stream.
        """
        word = unpack(self.endianness + ('H' if not self.signed else 'h'), self.readAt(self.offset, 2))[0]
        self.offset += 2
        return word