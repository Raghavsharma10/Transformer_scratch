def all_lemmas(self):
        '''
        A generator over all the lemmas in the GermaNet database.
        '''
        for lemma_dict in self._mongo_db.lexunits.find():
            yield Lemma(self, lemma_dict)