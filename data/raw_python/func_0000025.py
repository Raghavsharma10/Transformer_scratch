def gen_grid_2d(shape, voxelsize):
    """
    Generate list of edges for a base grid.
    """
    nr, nc = shape
    nrm1, ncm1 = nr - 1, nc - 1
    # sh = nm.asarray(shape)
    # calculate number of edges, in 2D: (nrows * (ncols - 1)) + ((nrows - 1) * ncols)
    nedges = 0
    for direction in range(len(shape)):
        sh = copy.copy(list(shape))
        sh[direction] += -1
        nedges += nm.prod(sh)

    nedges_old = ncm1 * nr + nrm1 * nc
    edges = nm.zeros((nedges, 2), dtype=nm.int16)
    edge_dir = nm.zeros((ncm1 * nr + nrm1 * nc,), dtype=nm.bool)
    nodes = nm.zeros((nm.prod(shape), 3), dtype=nm.float32)

    # edges
    idx = 0
    row = nm.zeros((ncm1, 2), dtype=nm.int16)
    row[:, 0] = nm.arange(ncm1)
    row[:, 1] = nm.arange(ncm1) + 1
    for ii in range(nr):
        edges[slice(idx, idx + ncm1), :] = row + nc * ii
        idx += ncm1

    edge_dir[slice(0, idx)] = 0  # horizontal dir

    idx0 = idx
    col = nm.zeros((nrm1, 2), dtype=nm.int16)
    col[:, 0] = nm.arange(nrm1) * nc
    col[:, 1] = nm.arange(nrm1) * nc + nc
    for ii in range(nc):
        edges[slice(idx, idx + nrm1), :] = col + ii
        idx += nrm1

    edge_dir[slice(idx0, idx)] = 1  # vertical dir

    # nodes
    idx = 0
    row = nm.zeros((nc, 3), dtype=nm.float32)
    row[:, 0] = voxelsize[0] * (nm.arange(nc) + 0.5)
    row[:, 1] = voxelsize[1] * 0.5
    for ii in range(nr):
        nodes[slice(idx, idx + nc), :] = row
        row[:, 1] += voxelsize[1]
        idx += nc

    return nodes, edges, edge_dir