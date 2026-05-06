def augmented_substructure(self, atoms, dante=False, deep=1, meta=False, as_view=True):
        """
        create substructure containing atoms and their neighbors

        :param atoms: list of core atoms in graph
        :param dante: if True return list of graphs containing atoms, atoms + first circle, atoms + 1st + 2nd,
            etc up to deep or while new nodes available
        :param deep: number of bonds between atoms and neighbors
        :param meta: copy metadata to each substructure
        :param as_view: If True, the returned graph-view provides a read-only view
            of the original graph without actually copying any data
        """
        nodes = [set(atoms)]
        for i in range(deep):
            n = {y for x in nodes[-1] for y in self._adj[x]} | nodes[-1]
            if n in nodes:
                break
            nodes.append(n)
        if dante:
            return [self.substructure(a, meta, as_view) for a in nodes]
        else:
            return self.substructure(nodes[-1], meta, as_view)