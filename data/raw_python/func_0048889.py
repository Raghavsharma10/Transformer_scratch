def dfs(node, expand=expansion_all, callback=None, silent=True):
    """ Perform a depth-first search on the node graph
    :param node: GraphNode
    :param expand: Returns the list of Nodes to explore from a Node
    :param callback: Callback to run in each node
    :param silent: Don't throw exception on circular dependency
    :return:
    """
    nodes = deque()
    for n in expand(node):
        nodes.append(n)

    while nodes:
        n = nodes.pop()
        n.visits += 1
        if callback:
            callback(n)
        for k in expand(n):
            if k.visits < 1:
                nodes.append(k)
            else:
                if not silent:
                    raise CircularDependency('Circular Dependency')