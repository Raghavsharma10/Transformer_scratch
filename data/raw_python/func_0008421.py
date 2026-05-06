def constraints(self, chunk):
        """ Returns a list of constraints that match the given Chunk.
        """
        a = [self._map1[w.index] for w in chunk.words if w.index in self._map1]
        b = []; [b.append(constraint) for constraint in a if constraint not in b]
        return b