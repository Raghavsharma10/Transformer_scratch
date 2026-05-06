def rownumber(self):
        """Read-only attribute providing the current 0-based index of the
        cursor in the result set or ``None`` if the index cannot be
        determined.

        The index can be seen as index of the cursor in a sequence
        (the result set). The next fetch operation will fetch the
        row indexed by :attr:`rownumber` in that sequence.
        """
        if self._frame is not None and self._pos is not None:
            return self._frame.offset + self._pos
        return self._pos