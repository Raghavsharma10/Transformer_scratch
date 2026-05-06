def get_mapping(self, other):
        """
        get self to other mapping
        """
        m = next(self._matcher(other).isomorphisms_iter(), None)
        if m:
            return {v: k for k, v in m.items()}