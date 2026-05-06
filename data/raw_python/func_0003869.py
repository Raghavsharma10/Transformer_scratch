def compute_rotsym(self, threshold=1e-3*angstrom):
        """Compute the rotational symmetry number.

           Optional argument:
            | ``threshold``  --  only when a rotation results in an rmsd below the given
                                 threshold, the rotation is considered to transform the
                                 molecule onto itself.
        """
        # Generate a graph with a more permissive threshold for bond lengths:
        # (is convenient in case of transition state geometries)
        graph = MolecularGraph.from_geometry(self, scaling=1.5)
        try:
            return compute_rotsym(self, graph, threshold)
        except ValueError:
            raise ValueError("The rotational symmetry number can only be computed when the graph is fully connected.")