def copy(self):
        """
        Return a shallow copy of a pqdict.

        """
        return self.__class__(self, key=self._keyfn, precedes=self._precedes)