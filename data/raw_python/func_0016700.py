def synsets(self, lemma, pos = None):
        '''
        Looks up synsets in the GermaNet database.

        Arguments:
        - `lemma`:
        - `pos`:
        '''
        return sorted(set(lemma_obj.synset
                          for lemma_obj in self.lemmas(lemma, pos)))