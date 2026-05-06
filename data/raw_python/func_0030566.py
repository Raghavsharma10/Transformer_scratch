def _make_query_from_terms(self, terms):
        """ Creates a query for dataset from decomposed search terms.

        Args:
            terms (dict or unicode or string):

        Returns:
            tuple: First element is str with FTS query, second is parameters of the query.

        """

        expanded_terms = self._expand_terms(terms)

        cterms = ''

        if expanded_terms['doc']:
            cterms = self.backend._and_join(expanded_terms['doc'])

        if expanded_terms['keywords']:
            if cterms:
                cterms = self.backend._and_join(
                    cterms, self.backend._join_keywords(expanded_terms['keywords']))
            else:
                cterms = self.backend._join_keywords(expanded_terms['keywords'])

        logger.debug('Dataset terms conversion: `{}` terms converted to `{}` query.'.format(terms, cterms))
        return cterms