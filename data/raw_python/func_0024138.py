def _matcher(self, other):
        """
        return VF2 GraphMatcher

        MoleculeContainer < MoleculeContainer
        MoleculeContainer < CGRContainer
        """
        if isinstance(other, (self._get_subclass('CGRContainer'), MoleculeContainer)):
            return GraphMatcher(other, self, lambda x, y: x == y, lambda x, y: x == y)
        raise TypeError('only cgr-cgr possible')