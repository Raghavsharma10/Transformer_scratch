def _flatten_descendants(self, include_parts=True):
        """Determines and stores all descendants of each GO term.

        Parameters
        ----------
        include_parts: bool, optional
            Whether to include ``part_of`` relations in determining
            descendants.

        Returns
        -------
        None
        """
        def get_all_descendants(term):
            descendants = set()
            for id_ in term.children:
                descendants.add(id_)
                descendants.update(get_all_descendants(self[id_]))
            if include_parts:
                for id_ in term.parts:
                    descendants.add(id_)
                    descendants.update(get_all_descendants(self[id_]))
            return descendants

        for term in self:
            term.descendants = get_all_descendants(term)