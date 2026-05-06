def _matcher(self, other):
        """
        QueryContainer < MoleculeContainer
        QueryContainer < QueryContainer[more general]
        QueryContainer < QueryCGRContainer[more general]
        """
        if isinstance(other, MoleculeContainer):
            return GraphMatcher(other, self, lambda x, y: y == x, lambda x, y: y == x)
        elif isinstance(other, (QueryContainer, QueryCGRContainer)):
            return GraphMatcher(other, self, lambda x, y: x == y, lambda x, y: x == y)
        raise TypeError('only query-molecule, query-query or query-cgr_query possible')