def label_tree(n,lookup):
    '''label tree will again recursively label the tree
    :param n: the root node, usually d3['children'][0]
    :param lookup: the node/id lookup
    '''
    if len(n["children"]) == 0:
        leaves = [lookup[n["node_id"]]]
    else:
        leaves = reduce(lambda ls, c: ls + label_tree(c,lookup), n["children"], [])
    del n["node_id"]
    n["name"] = name = "|||".join(sorted(map(str, leaves)))
    return leaves