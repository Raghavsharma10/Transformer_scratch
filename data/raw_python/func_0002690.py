def peekuntil(self, token, size=0):
        """
        Peeks for token into the FIFO.

        Performs the same function as readuntil() without removing data from the
        FIFO. See readuntil() for further information.
        """
        self.__append()

        i = self.buf.find(token, self.pos)
        if i < 0:
            index = max(len(token) - 1, size)
            newpos = max(len(self.buf) - index, self.pos)
            return False, self.buf[self.pos:newpos]

        newpos = i + len(token)
        return True, self.buf[self.pos:newpos]