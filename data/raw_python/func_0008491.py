def conjugate(self, verb, *args, **kwargs):
        """ Inflects the verb and returns the given tense (or None).
            For example: be
            - Verbs.conjugate("is", INFINITVE) => be
            - Verbs.conjugate("be", PRESENT, 1, SINGULAR) => I am
            - Verbs.conjugate("be", PRESENT, 1, PLURAL) => we are
            - Verbs.conjugate("be", PAST, 3, SINGULAR) => he was
            - Verbs.conjugate("be", PAST, aspect=PROGRESSIVE) => been
            - Verbs.conjugate("be", PAST, person=1, negated=True) => I wasn't
        """
        id = tense_id(*args, **kwargs)
        # Get the tense index from the format description (or a default).
        i1 = self._format.get(id)
        i2 = self._format.get(self._default.get(id))
        i3 = self._format.get(self._default.get(self._default.get(id)))
        b = self.lemma(verb, parse=kwargs.get("parse", True))
        v = []
        # Get the verb lexeme and return the requested index.
        if b in self:
            v = self[b]
            for i in (i1, i2, i3):
                if i is not None and 0 <= i < len(v) and v[i]:
                    return v[i]
        if kwargs.get("parse", True) is True: # rule-based
            v = self.find_lexeme(b)
            for i in (i1, i2, i3):
                if i is not None and 0 <= i < len(v) and v[i]:
                    return v[i]