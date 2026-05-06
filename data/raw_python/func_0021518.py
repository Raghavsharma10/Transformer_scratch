def writeByte(self, byte):
        """
        Writes a byte into the L{WriteData} stream object.
        
        @type byte: int
        @param byte: Byte value to write into the stream.
        """
        self.data.write(pack("B" if not self.signed else "b", byte))