def topological_sort(dependencies, start_nodes):
    """
    Perform a topological sort on the dependency graph `dependencies`, starting
    from list `start_nodes`.
    """
    retval = []
    def edges(node): return dependencies[node][1]
    def in_degree(node): return dependencies[node][0]
    def remove_incoming(node): dependencies[node][0] = in_degree(node) - 1
    while start_nodes:
        node = start_nodes.pop()
        retval.append(node)
        for child in edges(node):
            remove_incoming(child)
            if not in_degree(child):
                start_nodes.append(child)
    leftover_nodes = [node for node in list(dependencies.keys())
                      if in_degree(node) > 0]
    if leftover_nodes:
        raise CyclicDependency(leftover_nodes)
    else:
        return retval