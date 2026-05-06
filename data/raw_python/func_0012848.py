def prevnode(edges, component):
    """get the pervious component in the loop"""
    e = edges
    c = component
    n2c = [(a, b) for a, b in e if type(a) == tuple]
    c2n = [(a, b) for a, b in e if type(b) == tuple]
    node2cs = [(a, b) for a, b in e if b == c]
    c2nodes = []
    for node2c in node2cs:
        c2node = [(a, b) for a, b in c2n if b == node2c[0]]
        if len(c2node) == 0:
            # return []
            c2nodes = []
            break
        c2nodes.append(c2node[0])
    cs = [a for a, b in c2nodes]
    # test for connections that have no nodes
    # filter for no nodes
    nonodes = [(a, b) for a, b in e if type(a) != tuple and type(b) != tuple]
    for a, b in nonodes:
        if b == component:
            cs.append(a)
    return cs