def suggest(self, w):
        """ Return a list of (word, confidence) spelling corrections for the given word,
            based on the probability of known words with edit distance 1-2 from the given word.
        """
        if len(self) == 0:
            self.load()
        if len(w) == 1:
            return [(w, 1.0)] # I
        if w in PUNCTUATION:
            return [(w, 1.0)] # .?!
        if w.replace(".", "").isdigit():
            return [(w, 1.0)] # 1.5
        candidates = self._known([w]) \
                  or self._known(self._edit1(w)) \
                  or self._known(self._edit2(w)) \
                  or [w]
        candidates = [(self.get(c, 0.0), c) for c in candidates]
        s = float(sum(p for p, w in candidates) or 1)
        candidates = sorted(((p / s, w) for p, w in candidates), reverse=True)
        candidates = [(w.istitle() and x.title() or x, p) for p, x in candidates] # case-sensitive
        return candidates