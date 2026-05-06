def _get(self, word1, word2):
        """
        Return a possible next word after ``word1`` and ``word2``, or ``None``
        if there's no possibility.
        """
        key = self._WSEP.join([self._sanitize(word1), self._sanitize(word2)])
        key = key.lower()
        if key not in self._db:
            return

        return sample(self._db[key], 1)[0]