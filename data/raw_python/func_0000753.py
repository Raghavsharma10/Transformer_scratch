def get_static_parent(raml_resource, method=None):
    """ Get static parent resource of :raml_resource: with HTTP
    method :method:.

    :param raml_resource:Instance of ramlfications.raml.ResourceNode.
    :param method: HTTP method name which matching static resource
        must have.
    """
    parent = raml_resource.parent
    while is_dynamic_resource(parent):
        parent = parent.parent

    if parent is None:
        return parent

    match_method = method is not None
    if match_method:
        if parent.method.upper() == method.upper():
            return parent
    else:
        return parent

    for res in parent.root.resources:
        if res.path == parent.path:
            if res.method.upper() == method.upper():
                return res