def width_at_offset(self, n):
        """Returns the horizontal position of character n of the string"""
        #TODO make more efficient?
        width = wcswidth(self.s[:n])
        assert width != -1
        return width