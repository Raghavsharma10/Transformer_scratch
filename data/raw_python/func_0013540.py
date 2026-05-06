def getReadAlignments(self, reference, start=None, end=None):
        """
        Returns an iterator over the specified reads
        """
        return self._getReadAlignments(reference, start, end, self, None)