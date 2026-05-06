def cardinal_groupby(self):
        """
        Create an instance of this class for every step in the cardinal dimension.
        """
        if self._cardinal:
            g = self.network(fig=False)
            cardinal_indexes = self[self._cardinal].index.values
            selfs = {}
            cls = self.__class__
            for cardinal_index in cardinal_indexes:
                kwargs = {self._cardinal: self[self._cardinal].ix[[cardinal_index]]}
                for parent, child in nx.bfs_edges(g):
                    if child in kwargs:
                        continue
                    typ = g.edge_types[(parent, child)]
                    if self._cardinal in self[child].columns and hasattr(self[child], 'slice_cardinal'):
                        kwargs[child] = self[child].slice_cardinal(key)
                    elif typ == 'index-index':
                        # Select from the child on the parent's index (the parent is
                        # in the kwargs already).
                        kwargs[child] = self[child].ix[kwargs[parent].index.values]
                    elif typ == 'index-column':
                        # Select from the child where the column (of the same name as
                        # the parent) is in the parent's index values
                        cdf = self[child]
                        kwargs[child] = cdf[cdf[parent].isin(kwargs[parent].index.values)]
                    elif typ == 'column-index':
                        # Select from the child where the child's index is in the
                        # column of the parent. Note that this relationship
                        cdf = self[child]
                        cin = cdf.index.name
                        cols = [col for col in kwargs[parent] if cin == col or (cin == col[:-1] and col[-1].isdigit())]
                        index = kwargs[parent][cols].stack().astype(np.int64).values
                        kwargs[child] = cdf[cdf.index.isin(index)]
                selfs[cardinal_index] = cls(**kwargs)
        return selfs