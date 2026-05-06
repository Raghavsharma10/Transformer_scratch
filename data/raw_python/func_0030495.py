def _expand_terms(self, terms):
        """ Expands terms of the dataset to the appropriate fields. It will parse the search phrase
         and return only the search term components that are applicable to a Dataset query.

        Args:
            terms (dict or str):

        Returns:
            dict: keys are field names, values are query strings
        """

        ret = {
            'keywords': list(),
            'doc': list()}

        if not isinstance(terms, dict):
            stp = SearchTermParser()
            terms = stp.parse(terms, term_join=self.backend._and_join)

        if 'about' in terms:
            ret['doc'].append(terms['about'])

        if 'source' in terms:
            ret['keywords'].append(terms['source'])
        return ret