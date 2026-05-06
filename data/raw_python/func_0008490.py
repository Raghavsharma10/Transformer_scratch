def lexeme(self, verb, parse=True):
        """ Returns a list of all possible inflections of the given verb.
        """
        a = []
        b = self.lemma(verb, parse=parse)
        if b in self:
            a = [x for x in self[b] if x != ""]
        elif parse is True: # rule-based
            a = self.find_lexeme(b)
        u = []; [u.append(x) for x in a if x not in u]
        return u