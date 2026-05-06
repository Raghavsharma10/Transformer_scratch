def search(self, sentence):
        """ Returns a list of all matches found in the given sentence.
        """
        if sentence.__class__.__name__ == "Sentence":
            pass
        elif isinstance(sentence, list) or sentence.__class__.__name__ == "Text":
            a=[]; [a.extend(self.search(s)) for s in sentence]; return a
        elif isinstance(sentence, basestring):
            sentence = Sentence(sentence)
        elif isinstance(sentence, Match) and len(sentence) > 0:
            sentence = sentence[0].sentence.slice(sentence[0].index, sentence[-1].index + 1)
        a = []
        v = self._variations()
        u = {}
        m = self.match(sentence, _v=v)
        while m:
            a.append(m)
            m = self.match(sentence, start=m.words[-1].index+1, _v=v, _u=u)
        return a