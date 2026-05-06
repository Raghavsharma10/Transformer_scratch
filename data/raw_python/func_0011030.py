def unget_bytes(self, string):
        """Adds bytes to be internal buffer to be read

        This method is for reporting bytes from an in_stream read
        not initiated by this Input object"""

        self.unprocessed_bytes.extend(string[i:i + 1]
                                      for i in range(len(string)))