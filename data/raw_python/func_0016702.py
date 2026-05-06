def get_synset_by_id(self, mongo_id):
        '''
        Builds a Synset object from the database entry with the given
        ObjectId.

        Arguments:
        - `mongo_id`: a bson.objectid.ObjectId object
        '''
        cache_hit = None
        if self._synset_cache is not None:
            cache_hit = self._synset_cache.get(mongo_id)
        if cache_hit is not None:
            return cache_hit
        synset_dict = self._mongo_db.synsets.find_one({'_id': mongo_id})
        if synset_dict is not None:
            synset = Synset(self, synset_dict)
            if self._synset_cache is not None:
                self._synset_cache.put(mongo_id, synset)
            return synset