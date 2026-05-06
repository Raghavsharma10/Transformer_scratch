def substructure(self, atoms, meta=False, as_view=True):
        """
        create substructure containing atoms from nbunch list

        :param atoms: list of atoms numbers of substructure
        :param meta: if True metadata will be copied to substructure
        :param as_view: If True, the returned graph-view provides a read-only view
            of the original structure scaffold without actually copying any data.
        """
        s = self.subgraph(atoms)
        if as_view:
            s.add_atom = s.add_bond = s.delete_atom = s.delete_bond = frozen  # more informative exception
            return s
        s = s.copy()
        if not meta:
            s.graph.clear()
        return s