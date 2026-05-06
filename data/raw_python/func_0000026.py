def write_grid_to_vtk(fname, nodes, edges, node_flag=None, edge_flag=None):
    """
    Write nodes and edges to VTK file
    :param fname: VTK filename
    :param nodes:
    :param edges:
    :param node_flag: set if this node is really used in output
    :param edge_flag: set if this flag is used in output
    :return:
    """

    if node_flag is None:
        node_flag = np.ones([nodes.shape[0]], dtype=np.bool)
    if edge_flag is None:
        edge_flag = np.ones([edges.shape[0]], dtype=np.bool)
    nodes = make_nodes_3d(nodes)
    f = open(fname, "w")

    f.write("# vtk DataFile Version 2.6\n")
    f.write("output file\nASCII\nDATASET UNSTRUCTURED_GRID\n")

    idxs = nm.where(node_flag > 0)[0]
    nnd = len(idxs)
    aux = -nm.ones(node_flag.shape, dtype=nm.int32)
    aux[idxs] = nm.arange(nnd, dtype=nm.int32)
    f.write("\nPOINTS %d float\n" % nnd)
    for ndi in idxs:
        f.write("%.6f %.6f %.6f\n" % tuple(nodes[ndi, :]))

    idxs = nm.where(edge_flag > 0)[0]
    ned = len(idxs)
    f.write("\nCELLS %d %d\n" % (ned, ned * 3))
    for edi in idxs:
        f.write("2 %d %d\n" % tuple(aux[edges[edi, :]]))

    f.write("\nCELL_TYPES %d\n" % ned)
    for edi in idxs:
        f.write("3\n")