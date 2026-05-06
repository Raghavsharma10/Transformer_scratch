def _dfs_cycle_detect(graph, node, path, visited_nodes):
    """
    search graph for cycle using DFS continuing from node
    path contains the list of visited nodes currently on the stack
    visited_nodes is the set of already visited nodes
    :param graph:
    :param node:
    :param path:
    :param visited_nodes:
    :return:
    """
    visited_nodes.add(node)
    for target in graph[node]:
        if target in path:
            # cycle found => return current path
            return path + [target]
        else:
            return _dfs_cycle_detect(graph, target, path + [target], visited_nodes)
    return None