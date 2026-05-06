def write_newick(rootnode, 
	features=None, 
	format=1, 
	format_root_node=True,
    is_leaf_fn=None, 
    dist_formatter=None,
    support_formatter=None,
    name_formatter=None):
    """ 
    Iteratively export a tree structure and returns its NHX
    representation. 
    """
    newick = []
    leaf = is_leaf_fn if is_leaf_fn else lambda n: not bool(n.children)
    for postorder, node in rootnode.iter_prepostorder(is_leaf_fn=is_leaf_fn):
        if postorder:
            newick.append(")")
            if node.up is not None or format_root_node:
                newick.append(format_node(node, "internal", format,
                                          dist_formatter=dist_formatter,
                                          support_formatter=support_formatter,
                                          name_formatter=name_formatter))
                newick.append(_get_features_string(node, features))
        else:
            if node is not rootnode and node != node.up.children[0]:
                newick.append(",")

            if leaf(node):
                safe_name = re.sub("["+_ILEGAL_NEWICK_CHARS+"]", "_", \
                               str(getattr(node, "name")))
                newick.append(format_node(node, "leaf", format,
                              dist_formatter=dist_formatter,
                              support_formatter=support_formatter,
                              name_formatter=name_formatter))
                newick.append(_get_features_string(node, features))
            else:
                newick.append("(")

    newick.append(";")
    return ''.join(newick)