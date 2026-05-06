def iter_halfs_double(graph):
    """Select two random non-consecutive bonds that divide the molecule in two"""
    edges = graph.edges
    for index1, (atom_a1, atom_b1) in enumerate(edges):
        for atom_a2, atom_b2 in edges[:index1]:
            try:
                affected_atoms1, affected_atoms2, hinge_atoms = graph.get_halfs_double(atom_a1, atom_b1, atom_a2, atom_b2)
                yield affected_atoms1, affected_atoms2, hinge_atoms
            except GraphError:
                pass