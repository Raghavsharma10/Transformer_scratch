def search_model(self, q, text, lookup=None):
        '''Implements :meth:`stdnet.odm.SearchEngine.search_model`.
It return a new :class:`stdnet.odm.QueryElem` instance from
the input :class:`Query` and the *text* to search.'''
        words = self.words_from_text(text, for_search=True)
        if not words:
            return q
        qs = self._search(words, include=(q.model,), lookup=lookup)
        qs = tuple((q.get_field('object_id') for q in qs))
        return odm.intersect((q,)+qs)