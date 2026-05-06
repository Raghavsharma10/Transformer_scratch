def attr_subresource(raml_resource, route_name):
    """ Determine if :raml_resource: is an attribute subresource.

    :param raml_resource: Instance of ramlfications.raml.ResourceNode.
    :param route_name: Name of the :raml_resource:.
    """
    static_parent = get_static_parent(raml_resource, method='POST')
    if static_parent is None:
        return False
    schema = resource_schema(static_parent) or {}
    properties = schema.get('properties', {})
    if route_name in properties:
        db_settings = properties[route_name].get('_db_settings', {})
        return db_settings.get('type') in ('dict', 'list')
    return False