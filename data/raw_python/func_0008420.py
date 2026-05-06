def constraint(self, word):
        """ Returns the constraint that matches the given Word, or None.
        """
        if word.index in self._map1:
            return self._map1[word.index]