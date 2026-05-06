def all_independent_generators(self):
        """
        Return all generators in this namespace which are not clones.
        """
        return {g: name for g, name in self._ns.items() if not is_clone(g)}