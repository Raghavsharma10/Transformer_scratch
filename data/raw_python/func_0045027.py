def slice_cardinal(self, key):
        """
        Slice the container according to its (primary) cardinal axis.

        The "cardinal" axis can have any name so long as the name matches a
        data object attached to the container. The index name for this object
        should also match the value of the cardinal axis.

        The algorithm builds a network graph representing the data relationships
        (including information about the type of relationship) and then traverses
        the edge tree (starting from the cardinal table). Each subsequent child
        object in the tree is sliced based on its relationship with its parent.

        Note:
            Breadth first traversal is performed.

        Warning:
            This function does not make a copy (if possible): to ensure a new
            object is created (a copy) use :func:`~exa.core.container.Container.copy`
            after slicing.

            .. code-block:: Python

                myslice = mycontainer[::2].copy()

        See Also:
            For data network generation, see :func:`~exa.core.container.Container.network`.
            For information about relationships between data objects see
            :mod:`~exa.core.numerical`.
        """
        if self._cardinal:
            cls = self.__class__
            key = check_key(self[self._cardinal], key, cardinal=True)
            g = self.network(fig=False)
            kwargs = {self._cardinal: self[self._cardinal].ix[key], 'name': self.name,
                      'description': self.description, 'meta': self.meta}
            # Next traverse, breadth first, all data objects
            for parent, child in nx.bfs_edges(g, self._cardinal):
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
            return cls(**kwargs)