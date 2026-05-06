def search_queries(self, q):
        '''Return a new :class:`QueryElem` for *q* applying a text search.'''
        if self.text:
            searchengine = self.session.router.search_engine
            if searchengine:
                return searchengine.search_model(q, *self.text)
            else:
                raise QuerySetError('Search not available for %s' % self._meta)
        else:
            return q