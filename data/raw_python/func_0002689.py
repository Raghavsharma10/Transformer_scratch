def peekline(self):
        """
        Peeks a line into the FIFO.

        Perfroms the same function as readline() without removing data from the
        FIFO. See readline() for further information.
        """
        self.__append()

        i = self.buf.find(self.eol, self.pos)
        if i < 0:
            return ''

        newpos = i + len(self.eol)
        return self.buf[self.pos:newpos]