def readQword(self):
        """
        Reads a qword value from the L{ReadData} stream object.
        
        @rtype: int
        @return: The qword value read from the L{ReadData} stream.
        """
        qword = unpack(self.endianness + ('Q' if not self.signed else 'b'),  self.readAt(self.offset, 8))[0]
        self.offset += 8
        return qword