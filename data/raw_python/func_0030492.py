def _and_join(self, terms):
        """ Joins terms using AND operator.

        Args:
            terms (list): terms to join

        Examples:
            self._and_join(['term1']) -> 'term1'
            self._and_join(['term1', 'term2']) -> 'term1 AND term2'
            self._and_join(['term1', 'term2', 'term3']) -> 'term1 AND term2 AND term3'

        Returns:
            str
        """
        if len(terms) > 1:
            return ' AND '.join([self._or_join(t) for t in terms])
        else:
            return self._or_join(terms[0])