def get_resource_siblings(raml_resource):
    """ Get siblings of :raml_resource:.

    :param raml_resource: Instance of ramlfications.raml.ResourceNode.
    """
    path = raml_resource.path
    return [res for res in raml_resource.root.resources
            if res.path == path]