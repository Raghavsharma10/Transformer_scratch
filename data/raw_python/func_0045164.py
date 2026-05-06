def _slice(self, level):
        '''Return list of tuples of ident and lineage ident for given level
(numbered rank)'''
        if level >= len(self.taxonomy):
            raise IndexError('Level greater than size of taxonomy')
        res = []
        for ident in sorted(list(self.keys())):
            res.append((self[ident]['taxref'], self[ident]['lineage'][level]))
        return res