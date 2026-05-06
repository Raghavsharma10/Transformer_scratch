def beginning_of_line(self):
        r"""
        Return true if the scan pointer is at the beginning of a line.

            >>> s = Scanner("test\ntest\n")
            >>> s.beginning_of_line()
            True
            >>> s.skip(r'te')
            2
            >>> s.beginning_of_line()
            False
            >>> s.skip(r'st\n')
            3
            >>> s.beginning_of_line()
            True
            >>> s.terminate()
            >>> s.beginning_of_line()
            True
        """
        if self.pos > len(self.string):
            return None
        elif self.pos == 0:
            return True
        return self.string[self.pos - 1] == '\n'