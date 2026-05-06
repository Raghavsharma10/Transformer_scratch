def width_aware_splitlines(self, columns):
        # type: (int) -> Iterator[FmtStr]
        """Split into lines, pushing doublewidth characters at the end of a line to the next line.

        When a double-width character is pushed to the next line, a space is added to pad out the line.
        """
        if columns < 2:
            raise ValueError("Column width %s is too narrow." % columns)
        if wcswidth(self.s) == -1:
            raise ValueError('bad values for width aware slicing')
        return self._width_aware_splitlines(columns)