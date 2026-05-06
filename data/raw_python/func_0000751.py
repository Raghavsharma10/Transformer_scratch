def resource_view_attrs(raml_resource, singular=False):
    """ Generate view method names needed for `raml_resource` view.

    Collects HTTP method names from resource siblings and dynamic children
    if exist. Collected methods are then translated  to
    `nefertari.view.BaseView` method names, each of which is used to
    process a particular HTTP method request.

    Maps of {HTTP_method: view_method} `collection_methods` and
    `item_methods` are used to convert collection and item methods
    respectively.

    :param raml_resource: Instance of ramlfications.raml.ResourceNode
    :param singular: Boolean indicating if resource is singular or not
    """
    from .views import collection_methods, item_methods
    # Singular resource doesn't have collection methods though
    # it looks like a collection
    if singular:
        collection_methods = item_methods

    siblings = get_resource_siblings(raml_resource)
    http_methods = [sibl.method.lower() for sibl in siblings]
    attrs = [collection_methods.get(method) for method in http_methods]

    # Check if resource has dynamic child resource like collection/{id}
    # If dynamic child resource exists, add its siblings' methods to attrs,
    # as both resources are handled by a single view
    children = get_resource_children(raml_resource)
    http_submethods = [child.method.lower() for child in children
                       if is_dynamic_uri(child.path)]
    attrs += [item_methods.get(method) for method in http_submethods]

    return set(filter(bool, attrs))