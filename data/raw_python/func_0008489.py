def lemma(self, verb, parse=True):
        """ Returns the infinitive form of the given verb, or None.
        """
        if dict.__len__(self) == 0:
            self.load()
        if verb.lower() in self._inverse:
            return self._inverse[verb.lower()]
        if verb in self._inverse:
            return self._inverse[verb]
        if parse is True: # rule-based
            return self.find_lemma(verb)