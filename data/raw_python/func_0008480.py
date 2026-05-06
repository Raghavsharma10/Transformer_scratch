def apply(self, tokens):
        """ Applies the named entity recognizer to the given list of tokens,
            where each token is a [word, tag] list.
        """
        # Note: we could also scan for patterns, e.g.,
        # "my|his|her name is|was *" => NNP-PERS.
        i = 0
        while i < len(tokens):
            w = tokens[i][0].lower()
            if RE_ENTITY1.match(w) \
            or RE_ENTITY2.match(w) \
            or RE_ENTITY3.match(w):
                tokens[i][1] = self.tag
            if w in self:
                for e in self[w]:
                    # Look ahead to see if successive words match the named entity.
                    e, tag = (e[:-1], "-"+e[-1].upper()) if e[-1] in self._cmd else (e, "")
                    b = True
                    for j, e in enumerate(e):
                        if i + j >= len(tokens) or tokens[i+j][0].lower() != e:
                            b = False; break
                    if b:
                        for token in tokens[i:i+j+1]:
                            token[1] = token[1] if token[1].startswith(self.tag) else self.tag
                            token[1] += tag
                        i += j
                        break
            i += 1
        return tokens