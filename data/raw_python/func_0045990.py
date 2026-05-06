def getch(self):
        """
        Get a single character and advance the scan pointer.

            >>> s = Scanner("abc")
            >>> s.getch()
            'a'
            >>> s.getch()
            'b'
            >>> s.getch()
            'c'
            >>> s.pos
            3
        """
        self.pos += 1
        return self.string[self.pos - 1:self.pos]