def writeDword(self, dword):
        """
        Writes a dword value into the L{WriteData} stream object.
        
        @type dword: int
        @param dword: Dword value to write into the stream.
        """        
        self.data.write(pack(self.endianness + ("L" if not self.signed else "l"), dword))