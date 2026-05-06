def get_lemma_by_id(self, mongo_id):
        '''
        Builds a Lemma object from the database entry with the given
        ObjectId.

        Arguments:
        - `mongo_id`: a bson.objectid.ObjectId object
        '''
        cache_hit = None
        if self._lemma_cache is not None:
            cache_hit = self._lemma_cache.get(mongo_id)
        if cache_hit is not None:
            return cache_hit
        lemma_dict = self._mongo_db.lexunits.find_one({'_id': mongo_id})
        if lemma_dict is not None:
            lemma = Lemma(self, lemma_dict)
            if self._lemma_cache is not None:
                self._lemma_cache.put(mongo_id, lemma)
            return lemma