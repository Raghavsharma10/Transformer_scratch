def all_synsets(self):
        '''
        A generator over all the synsets in the GermaNet database.
        '''
        for synset_dict in self._mongo_db.synsets.find():
            yield Synset(self, synset_dict)