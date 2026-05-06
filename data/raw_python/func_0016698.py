def lemmas(self, lemma, pos = None):
        '''
        Looks up lemmas in the GermaNet database.

        Arguments:
        - `lemma`:
        - `pos`:
        '''
        if pos is not None:
            if pos not in SHORT_POS_TO_LONG:
                return None
            pos         = SHORT_POS_TO_LONG[pos]
            lemma_dicts = self._mongo_db.lexunits.find({'orthForm': lemma,
                                                        'category': pos})
        else:
            lemma_dicts = self._mongo_db.lexunits.find({'orthForm': lemma})
        return sorted([Lemma(self, lemma_dict) for lemma_dict in lemma_dicts])