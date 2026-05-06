def resource_of_node(resources, node):
    """ Returns resource of node.
    """
    for resource in resources:
        model = getattr(resource, 'model', None)
        if type(node) == model:
            return resource
    return BasePageResource