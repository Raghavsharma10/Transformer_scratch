def constituents(self, constraint=None):
        """ Returns a list of Word and Chunk objects, 
            where words have been grouped into their chunks whenever possible.
            Optionally, returns only chunks/words that match given constraint(s), or constraint index.
        """
        # Select only words that match the given constraint.
        # Note: this will only work with constraints from Match.pattern.sequence.
        W = self.words
        n = len(self.pattern.sequence)
        if isinstance(constraint, (int, Constraint)):
            if isinstance(constraint, int):
                i = constraint 
                i = i<0 and i%n or i
            else:
                i = self.pattern.sequence.index(constraint)
            W = self._map2.get(i,[])
            W = [self.words[i-self.words[0].index] for i in W]            
        if isinstance(constraint, (list, tuple)):
            W = []; [W.extend(self._map2.get(j<0 and j%n or j,[])) for j in constraint]
            W = [self.words[i-self.words[0].index] for i in W]
            W = unique(W)
        a = []
        i = 0
        while i < len(W):
            w = W[i]
            if w.chunk and W[i:i+len(w.chunk)] == w.chunk.words:
                i += len(w.chunk) - 1
                a.append(w.chunk)
            else:
                a.append(w)
            i += 1
        return a