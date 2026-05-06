def _straight_line_vertices(adjacency_mat, node_coords, directed=False):
    """
    Generate the vertices for straight lines between nodes.

    If it is a directed graph, it also generates the vertices which can be
    passed to an :class:`ArrowVisual`.

    Parameters
    ----------
    adjacency_mat : array
        The adjacency matrix of the graph
    node_coords : array
        The current coordinates of all nodes in the graph
    directed : bool
        Wether the graph is directed. If this is true it will also generate
        the vertices for arrows which can be passed to :class:`ArrowVisual`.

    Returns
    -------
    vertices : tuple
        Returns a tuple containing containing (`line_vertices`,
        `arrow_vertices`)
    """

    if not issparse(adjacency_mat):
        adjacency_mat = np.asarray(adjacency_mat, float)

    if (adjacency_mat.ndim != 2 or adjacency_mat.shape[0] !=
            adjacency_mat.shape[1]):
        raise ValueError("Adjacency matrix should be square.")

    arrow_vertices = np.array([])

    edges = _get_edges(adjacency_mat)
    line_vertices = node_coords[edges.ravel()]

    if directed:
        arrows = np.array(list(_get_directed_edges(adjacency_mat)))
        arrow_vertices = node_coords[arrows.ravel()]
        arrow_vertices = arrow_vertices.reshape((len(arrow_vertices)/2, 4))

    return line_vertices, arrow_vertices