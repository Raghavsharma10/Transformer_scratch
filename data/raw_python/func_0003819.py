def setup_ics(graph):
    """Make a list of internal coordinates based on the graph

       Argument:
        | ``graph`` -- A Graph instance.

       The list of internal coordinates will include all bond lengths, all
       bending angles, and all dihedral angles.
    """
    ics = []
    # A) Collect all bonds.
    for i0, i1 in graph.edges:
        ics.append(BondLength(i0, i1))
    # B) Collect all bends. (see b_bending_angles.py for the explanation)
    for i1 in range(graph.num_vertices):
        n = list(graph.neighbors[i1])
        for index, i0 in enumerate(n):
            for i2 in n[:index]:
                ics.append(BendingAngle(i0, i1, i2))
    # C) Collect all dihedrals.
    for i1, i2 in graph.edges:
        for i0 in graph.neighbors[i1]:
            if i0==i2:
                # All four indexes must be different.
                continue
            for i3 in graph.neighbors[i2]:
                if i3==i1 or i3==i0:
                    # All four indexes must be different.
                    continue
                ics.append(DihedralAngle(i0, i1, i2, i3))
    return ics