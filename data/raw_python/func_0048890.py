def explore(node):
    """ Given a node, explores on relatives, siblings and children
    :param node: GraphNode from which to explore
    :return: set of explored GraphNodes
    """
    explored = set()
    explored.add(node)
    dfs(node, callback=lambda n: explored.add(n))
    return explored