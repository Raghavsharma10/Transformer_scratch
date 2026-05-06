def writeWord(self, word):
        """
        Writes a word value into the L{WriteData} stream object.
        
        @type word: int
        @param word: Word value to write into the stream.
        """
        self.data.write(pack(self.endianness + ("H" if not self.signed else "h"), word))