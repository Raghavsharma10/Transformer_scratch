def edges2nodes(edges):
    """gather the nodes from the edges"""
    nodes = []
    for e1, e2 in edges:
        nodes.append(e1)
        nodes.append(e2)
    nodedict = dict([(n, None) for n in nodes])
    justnodes = list(nodedict.keys())
    # justnodes.sort()
    justnodes = sorted(justnodes, key=lambda x: str(x[0]))
    return justnodes