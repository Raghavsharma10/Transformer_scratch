def create_edges(cells_nodes):
    """Setup edge-node and edge-cell relations. Adapted from voropy.
    """
    # Create the idx_hierarchy (nodes->edges->cells), i.e., the value of
    # `self.idx_hierarchy[0, 2, 27]` is the index of the node of cell 27, edge
    # 2, node 0. The shape of `self.idx_hierarchy` is `(2, 3, n)`, where `n` is
    # the number of cells. Make sure that the k-th edge is opposite of the k-th
    # point in the triangle.
    local_idx = numpy.array([[1, 2], [2, 0], [0, 1]]).T
    # Map idx back to the nodes. This is useful if quantities which are in
    # idx shape need to be added up into nodes (e.g., equation system rhs).
    nds = cells_nodes.T
    idx_hierarchy = nds[local_idx]

    s = idx_hierarchy.shape
    a = numpy.sort(idx_hierarchy.reshape(s[0], s[1] * s[2]).T)

    b = numpy.ascontiguousarray(a).view(
        numpy.dtype((numpy.void, a.dtype.itemsize * a.shape[1]))
    )
    _, idx, inv, cts = numpy.unique(
        b, return_index=True, return_inverse=True, return_counts=True
    )

    # No edge has more than 2 cells. This assertion fails, for example, if
    # cells are listed twice.
    assert all(cts < 3)

    edge_nodes = a[idx]
    cells_edges = inv.reshape(3, -1).T

    return edge_nodes, cells_edges