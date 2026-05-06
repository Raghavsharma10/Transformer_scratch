def iteritems(self):
        """
        Iterates over all mappings


        Yields
        ------
        (int,Mapping)
            The next pair (index, mapping)
        """
        for m in self.mappings:
            yield self.indexes[m.clause][m.target], m