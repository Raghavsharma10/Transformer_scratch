def tenses(self, verb, parse=True):
        """ Returns a list of possible tenses for the given inflected verb.
        """
        verb = verb.lower()
        a = set()
        b = self.lemma(verb, parse=parse)
        v = []
        if b in self:
            v = self[b]
        elif parse is True: # rule-based
            v = self.find_lexeme(b)
        # For each tense in the verb lexeme that matches the given tense,
        # 1) retrieve the tense tuple,
        # 2) retrieve the tense tuples for which that tense is a default.
        for i, tense in enumerate(v):
            if tense == verb:
                for id, index in self._format.items():
                    if i == index:
                        a.add(id)
                for id1, id2 in self._default.items():
                    if id2 in a:
                        a.add(id1)
                for id1, id2 in self._default.items():
                    if id2 in a:
                        a.add(id1)
        a = (TENSES[id][:-2] for id in a)
        a = Tenses(sorted(a))
        return a