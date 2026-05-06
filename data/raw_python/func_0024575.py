def substructure(self, atoms, meta=False, as_view=True):
        """
        create substructure containing atoms from nbunch list

        :param atoms: list of atoms numbers of substructure
        :param meta: if True metadata will be copied to substructure
        :param as_view: If True, the returned graph-view provides a read-only view
            of the original structure scaffold without actually copying any data
        """
        s = super().substructure(atoms, meta, as_view)
        if as_view:
            s.reset_query_marks = frozen
        return s