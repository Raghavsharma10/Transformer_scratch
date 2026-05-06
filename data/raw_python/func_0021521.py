def writeQword(self, qword):
        """
        Writes a qword value into the L{WriteData} stream object.
        
        @type qword: int
        @param qword: Qword value to write into the stream.
        """
        self.data.write(pack(self.endianness + ("Q" if not self.signed else "q"),  qword))