def readuntil(self, token, size=0):
        """
        Reads data from the FIFO until a token is encountered.

        If no token is encountered as much data is read from the FIFO as
        possible keeping in mind that the FIFO must retain enough data to
        perform matches for the token across writes.

        Args:
            token: The token to read until.
            size: The minimum amount of data that should be left in the FIFO.
                This is only used if it is greater than the length of the
                token.  When ommited this value will default to the length of
                the token.

        Returns: A tuple of (found, data) where found is a boolean indicating
            whether the token was found, and data is all the data that could be
            read from the FIFO.

        Note: When a token is found the token is also read from the buffer and
            returned in the data.
        """
        self.__append()

        i = self.buf.find(token, self.pos)
        if i < 0:
            index = max(len(token) - 1, size)
            newpos = max(len(self.buf) - index, self.pos)
            data = self.buf[self.pos:newpos]
            self.pos = newpos
            self.__discard()
            return False, data

        newpos = i + len(token)
        data = self.buf[self.pos:newpos]
        self.pos = newpos
        self.__discard()
        return True, data