def readString(self):
        """
        Reads an ASCII string from the L{ReadData} stream object.
        
        @rtype: str
        @return: An ASCII string read form the stream.
        """
        resultStr = ""
        while self.data[self.offset] != "\x00":
            resultStr += self.data[self.offset]
            self.offset += 1
        return resultStr