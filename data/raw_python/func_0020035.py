def _search(self, words, include=None, exclude=None, lookup=None):
        '''Full text search. Return a list of queries to intersect.'''
        lookup = lookup or 'contains'
        query = self.router.worditem.query()
        if include:
            query = query.filter(model_type__in=include)
        if exclude:
            query = query.exclude(model_type__in=include)
        if not words:
            return [query]
        qs = []
        if lookup == 'in':
            # we are looking for items with at least one word in it
            qs.append(query.filter(word__in=words))
        elif lookup == 'contains':
            #we want to match every single words
            for word in words:
                qs.append(query.filter(word=word))
        else:
            raise ValueError('Unknown lookup "{0}"'.format(lookup))
        return qs