def _or_join(self, terms):
        """ Joins terms using OR operator.

        Args:
            terms (list): terms to join

        Examples:
            self._or_join(['term1', 'term2']) -> 'term1 OR term2'

        Returns:
            str
        """

        if isinstance(terms, (tuple, list)):
            if len(terms) > 1:
                return '(' + ' OR '.join(terms) + ')'
            else:
                return terms[0]
        else:
            return terms