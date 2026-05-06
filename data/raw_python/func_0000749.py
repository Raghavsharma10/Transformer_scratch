def dynamic_part_name(raml_resource, route_name, pk_field):
    """ Generate a dynamic part for a resource :raml_resource:.

    A dynamic part is generated using 2 parts: :route_name: of the
    resource and the dynamic part of first dynamic child resources. If
    :raml_resource: has no dynamic child resources, 'id' is used as the
    2nd part.
    E.g. if your dynamic part on route 'stories' is named 'superId' then
    dynamic part will be 'stories_superId'.

    :param raml_resource: Instance of ramlfications.raml.ResourceNode for
        which dynamic part name is being generated.
    :param route_name: Cleaned name of :raml_resource:
    :param pk_field: Model Primary Key field name.
    """
    subresources = get_resource_children(raml_resource)
    dynamic_uris = [res.path for res in subresources
                    if is_dynamic_uri(res.path)]
    if dynamic_uris:
        dynamic_part = extract_dynamic_part(dynamic_uris[0])
    else:
        dynamic_part = pk_field
    return '_'.join([route_name, dynamic_part])