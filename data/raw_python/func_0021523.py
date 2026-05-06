def skipBytes(self, nroBytes):
        """
        Skips the specified number as parameter to the current value of the L{WriteData} stream.
        
        @type nroBytes: int
        @param nroBytes: The number of bytes to skip.
        """
        self.data.seek(nroBytes + self.data.tell())