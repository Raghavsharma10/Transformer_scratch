def get_resource_children(raml_resource):
    """ Get children of :raml_resource:.

    :param raml_resource: Instance of ramlfications.raml.ResourceNode.
    """
    path = raml_resource.path
    return [res for res in raml_resource.root.resources
            if res.parent and res.parent.path == path]