def bin(self):
        """The bin index of this mark.

        :returns: An integer bin index or None if the mark is inactive.

        """
        bin = self._query(('MBIN?', Integer, Integer), self.idx)
        return None if bin == -1 else bin