def iter_halfs_bend(graph):
    """Select randomly two consecutive bonds that divide the molecule in two"""
    for atom2 in range(graph.num_vertices):
        neighbors = list(graph.neighbors[atom2])
        for index1, atom1 in enumerate(neighbors):
            for atom3 in neighbors[index1+1:]:
                try:
                    affected_atoms = graph.get_halfs(atom2, atom1)[0]
                    # the affected atoms never contain atom1!
                    yield affected_atoms, (atom1, atom2, atom3)
                    continue
                except GraphError:
                    pass
                try:
                    affected_atoms = graph.get_halfs(atom2, atom3)[0]
                    # the affected atoms never contain atom3!
                    yield affected_atoms, (atom3, atom2, atom1)
                except GraphError:
                    pass