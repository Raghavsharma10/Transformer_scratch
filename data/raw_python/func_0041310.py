def detect_cycle(graph):
    """
    search the given directed graph for cycles

    returns None if the given graph is cycle free
    otherwise it returns a path through the graph that contains a cycle
    :param graph:
    :return:
    """

    visited_nodes = set()

    for node in list(graph):
        if node not in visited_nodes:
            cycle = _dfs_cycle_detect(graph, node, [node], visited_nodes)
            if cycle:
                return cycle
    return None