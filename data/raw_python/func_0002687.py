def read(self, length=-1):
        """
        Reads from the FIFO.

        Reads as much data as possible from the FIFO up to the specified
        length. If the length argument is negative or ommited all data
        currently available in the FIFO will be read. If there is no data
        available in the FIFO an empty string is returned.

        Args:
            length: The amount of data to read from the FIFO. Defaults to -1.
        """
        if 0 <= length < len(self):
            newpos = self.pos + length
            data = self.buf[self.pos:newpos]
            self.pos = newpos
            self.__discard()
            return data

        data = self.buf[self.pos:]
        self.clear()
        return data