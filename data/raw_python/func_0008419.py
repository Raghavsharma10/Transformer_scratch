def match(self, sentence, start=0, _v=None, _u=None):
        """ Returns the first match found in the given sentence, or None.
        """
        if sentence.__class__.__name__ == "Sentence":
            pass
        elif isinstance(sentence, list) or sentence.__class__.__name__ == "Text":
            return find(lambda m,s: m is not None, ((self.match(s, start, _v), s) for s in sentence))[0]
        elif isinstance(sentence, basestring):
            sentence = Sentence(sentence)
        elif isinstance(sentence, Match) and len(sentence) > 0:
            sentence = sentence[0].sentence.slice(sentence[0].index, sentence[-1].index + 1)
        # Variations (_v) further down the list may match words more to the front.
        # We need to check all of them. Unmatched variations are blacklisted (_u).
        # Pattern.search() calls Pattern.match() with a persistent blacklist (1.5x faster).
        a = []
        for sequence in (_v is not None and _v or self._variations()):
            if _u is not None and id(sequence) in _u:
                continue
            m = self._match(sequence, sentence, start)
            if m is not None:
                a.append((m.words[0].index, len(m.words), m))
            if m is not None and m.words[0].index == start:
                return m
            if m is None and _u is not None:
                _u[id(sequence)] = False
        # Return the leftmost-longest.
        if len(a) > 0:
            return sorted(a)[0][-1]