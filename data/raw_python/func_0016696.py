def cache_size(self, new_value):
        '''
        Set the cache size used to reduce the number of database
        access operations.
        '''
        if type(new_value) == int and 0 < new_value:
            if self._lemma_cache is not None:
                self._lemma_cache  = repoze.lru.LRUCache(new_value)
                self._synset_cache = repoze.lru.LRUCache(new_value)