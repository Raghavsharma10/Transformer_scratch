def _matcher(self, other):
        """
        QueryCGRContainer < CGRContainer
        QueryContainer < QueryCGRContainer[more general]
        """
        if isinstance(other, CGRContainer):
            return GraphMatcher(other, self, lambda x, y: y == x, lambda x, y: y == x)
        elif isinstance(other, QueryCGRContainer):
            return GraphMatcher(other, self, lambda x, y: x == y, lambda x, y: x == y)
        raise TypeError('only cgr_query-cgr or cgr_query-cgr_query possible')