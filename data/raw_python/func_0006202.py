def graph_from_edges(edge_list, node_prefix='', directed=False):
    """Creates a basic graph out of an edge list.

    The edge list has to be a list of tuples representing
    the nodes connected by the edge.
    The values can be anything: bool, int, float, str.

    If the graph is undirected by default, it is only
    calculated from one of the symmetric halves of the matrix.
    """
    if edge_list is None:
        edge_list = []

    graph_type = "digraph" if directed else "graph"
    with_prefix = functools.partial("{0}{1}".format, node_prefix)

    graph = Dot(graph_type=graph_type)

    for src, dst in edge_list:
        src = with_prefix(src)
        dst = with_prefix(dst)

        graph.add_edge(Edge(src, dst))

    return graph