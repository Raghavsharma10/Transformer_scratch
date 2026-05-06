def _contextualise(self):
        '''Determine contextual idents (cidents)'''
        # loop through hierarchy identifying unique lineages
        # TODO: gain other contextual information, not just ident
        deja_vues = []
        for rank in reversed(self.taxonomy):
            # return named clades -- '' are ignored
            clades = [e for e in self.hierarchy[rank] if e[1]]
            # print 'Rank: {0} - {1}'.format(rank, len(clades))
            # get unique lineages at this level
            uniques = [e for e in clades if len(e[0]) == 1]
            # removed those already seen
            uniques = [e for e in uniques if e[0][0].ident not in deja_vues]
            # add each to self[ident]['cident']
            for e in uniques:
                ident = e[0][0].ident
                self[ident]['cident'] = e[1]
                deja_vues.append(ident)