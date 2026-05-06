def _translate_nodes(root, *nodes):
    """
    Convert node names into node instances...
    """
    #name2node = {[n, None] for n in nodes if type(n) is str}
    name2node = dict([[n, None] for n in nodes if type(n) is str])
    for n in root.traverse():
        if n.name in name2node:
            if name2node[n.name] is not None:
                raise TreeError("Ambiguous node name: {}".format(str(n.name)))
            else:
                name2node[n.name] = n

    if None in list(name2node.values()):
        notfound = [key for key, value in six.iteritems(name2node) if value is None]
        raise ValueError("Node names not found: "+str(notfound))

    valid_nodes = []
    for n in nodes:
        if type(n) is not str:
            if type(n) is not root.__class__:
                raise TreeError("Invalid target node: "+str(n))
            else:
                valid_nodes.append(n)

    valid_nodes.extend(list(name2node.values()))
    if len(valid_nodes) == 1:
        return valid_nodes[0]
    else:
        return valid_nodes