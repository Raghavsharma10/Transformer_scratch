def is_method_call(node, method_name):
    """
    Returns True if `node` is a method call for `method_name`. `method_name`
    can be either a string or an iterable of strings.
    """

    if not isinstance(node, nodes.Call):
        return False

    if isinstance(node.node, nodes.Getattr):
        # e.g. foo.bar()
        method = node.node.attr

    elif isinstance(node.node, nodes.Name):
        # e.g. bar()
        method = node.node.name

    elif isinstance(node.node, nodes.Getitem):
        # e.g. foo["bar"]()
        method = node.node.arg.value

    else:
        return False

    if isinstance(method_name, (list, tuple)):
        return method in method_name

    return method == method_name