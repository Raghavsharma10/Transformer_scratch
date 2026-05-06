def _expand_terms(self, terms):
        """ Expands partition terms to the appropriate fields.

        Args:
            terms (dict or str):

        Returns:
            dict: keys are field names, values are query strings
        """
        ret = {
            'keywords': list(),
            'doc': list(),
            'from': None,
            'to': None}

        if not isinstance(terms, dict):
            stp = SearchTermParser()
            terms = stp.parse(terms, term_join=self.backend._and_join)

        if 'about' in terms:
            ret['doc'].append(terms['about'])

        if 'with' in terms:
            ret['doc'].append(terms['with'])

        if 'in' in terms:
            place_vids = self._expand_place_ids(terms['in'])
            ret['keywords'].append(place_vids)

        if 'by' in terms:
            ret['keywords'].append(terms['by'])
        ret['from'] = terms.get('from', None)
        ret['to'] = terms.get('to', None)
        return ret