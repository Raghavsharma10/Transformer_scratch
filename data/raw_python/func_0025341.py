def print_tree_recursive(tree_obj, node_index, attribute_names=None):
    """
    Recursively writes a string representation of a decision tree object.

    Parameters
    ----------
    tree_obj : sklearn.tree._tree.Tree object
        A base decision tree object
    node_index : int
        Index of the node being printed
    attribute_names : list
        List of attribute names
    
    Returns
    -------
    tree_str : str
        String representation of decision tree in the same format as the parf library.
    """
    tree_str = ""
    if node_index == 0:
        tree_str += "{0:d}\n".format(tree_obj.node_count)
    if tree_obj.feature[node_index] >= 0:
        if attribute_names is None:
            attr_val = "{0:d}".format(tree_obj.feature[node_index])
        else:
            attr_val = attribute_names[tree_obj.feature[node_index]]
        tree_str += "b {0:d} {1} {2:0.4f} {3:d} {4:1.5e}\n".format(node_index,
                                                                   attr_val,
                                                                   tree_obj.weighted_n_node_samples[node_index],
                                                                   tree_obj.n_node_samples[node_index],
                                                                   tree_obj.threshold[node_index])
    else:
        if tree_obj.max_n_classes > 1:
            leaf_value = "{0:d}".format(tree_obj.value[node_index].argmax())
        else:
            leaf_value = "{0}".format(tree_obj.value[node_index][0][0])
        tree_str += "l {0:d} {1} {2:0.4f} {3:d}\n".format(node_index,
                                                          leaf_value,
                                                          tree_obj.weighted_n_node_samples[node_index],
                                                          tree_obj.n_node_samples[node_index])
    if tree_obj.children_left[node_index] > 0:
        tree_str += print_tree_recursive(tree_obj, tree_obj.children_left[node_index], attribute_names)
    if tree_obj.children_right[node_index] > 0:
        tree_str += print_tree_recursive(tree_obj, tree_obj.children_right[node_index], attribute_names)
    return tree_str