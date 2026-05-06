def tree_view(dictionary, level=0, sep="|  "):
    """
    View a dictionary as a tree.
    """
    return "".join(["{0}{1}\n{2}".format(sep * level, k,
                   tree_view(v, level + 1, sep=sep) if isinstance(v, dict)
                   else "") for k, v in dictionary.items()])