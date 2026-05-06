def _hierarchy(self):
        '''Generate dictionary of referenced idents grouped by shared rank'''
        self.hierarchy = {}
        for rank in self.taxonomy:
            # extract lineage idents for this rank
            taxslice = self._slice(level=self.taxonomy.index(rank))
            # group idents by shared group at this rank
            self.hierarchy[rank] = self._group(taxslice)