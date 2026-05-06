def width(self):
        """The number of columns it would take to display this string"""
        if self._width is not None:
            return self._width
        self._width = sum(fs.width for fs in self.chunks)
        return self._width