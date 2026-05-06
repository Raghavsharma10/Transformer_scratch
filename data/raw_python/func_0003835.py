def iter_halfs_bond(graph):
    """Select a random bond (pair of atoms) that divides the molecule in two"""
    for atom1, atom2 in graph.edges:
        try:
            affected_atoms1, affected_atoms2 = graph.get_halfs(atom1, atom2)
            yield affected_atoms1, affected_atoms2, (atom1, atom2)
        except GraphError:
            # just try again
            continue