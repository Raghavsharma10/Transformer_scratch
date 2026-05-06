def _edit2(self, w):
        """ Returns a set of words with edit distance 2 from the given word
        """
        # Of all spelling errors, 99% is covered by edit distance 2.
        # Only keep candidates that are actually known words (20% speedup).
        return set(e2 for e1 in self._edit1(w) for e2 in self._edit1(e1) if e2 in self)