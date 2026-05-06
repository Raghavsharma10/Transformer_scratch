def _matcher(self, other):
        """
        CGRContainer < CGRContainer
        """
        if isinstance(other, CGRContainer):
            return GraphMatcher(other, self, lambda x, y: x == y, lambda x, y: x == y)
        raise TypeError('only cgr-cgr possible')