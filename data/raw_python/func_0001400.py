def new_concept(self, tag, clemma="", tokens=None, cidx=None, **kwargs):
        ''' Create a new concept object and add it to concept list
        tokens can be a list of Token objects or token indices
        '''
        if cidx is None:
            cidx = self.new_concept_id()
        if tokens:
            tokens = (t if isinstance(t, Token) else self[t] for t in tokens)
        c = Concept(cidx=cidx, tag=tag, clemma=clemma, sent=self, tokens=tokens, **kwargs)
        return self.add_concept(c)