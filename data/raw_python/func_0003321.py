def outputdata(self, data):
        """
        Send output with fixed length data
        """
        if not isinstance(data, bytes):
            data = str(data).encode(self.encoding)
        self.output(MemoryStream(data))