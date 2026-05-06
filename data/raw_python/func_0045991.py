def peek(self, length):
        """
        Get a number of characters without advancing the scan pointer.

            >>> s = Scanner("test string")
            >>> s.peek(7)
            'test st'
            >>> s.peek(7)
            'test st'
        """
        return self.string[self.pos:self.pos + length]