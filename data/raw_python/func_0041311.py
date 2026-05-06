def topsort(graph):
    """
    For the given graph, returns a list of nodes in topological order
    In py3 the behaviour of this function differs from py2,
    the resulting order will change with every execution in py3
    while in py2 the order stays the same
    :param graph:
    :return:
    """

    count = defaultdict(int)
    for feature, node in graph.items():
        for target in node:
            count[target] += 1
    # convert for list is necessary for py3 as in py3 the filter
    # function creates a filter object, in py2 it returns a list
    free_nodes = list(filter(lambda x: count[x] == 0, graph))
    result = []
    while free_nodes:
        node = free_nodes.pop()
        result.append(node)
        for target in graph[node]:
            count[target] -= 1
            if count[target] == 0:
                free_nodes.append(target)
    return result