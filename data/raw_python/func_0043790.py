def url(route, resource_id=None, pagination=None, **parameters):
    """
    Generates an absolute URL to an API resource.

    :param route: One of the routes available (see the header of this file)
    :type route: string
    :param resource_id: The resource ID you want. If None, it will point to the endpoint.
    :type resource_id: string|None
    :param pagination: parameters for pagination
    :type pagination: dict|None
    :param parameters: additional parameters required by the route

    :return the absolute route to the API
    :rtype string
    """
    route = route.format(**parameters)

    resource_id_url = '/' + str(resource_id) if resource_id else ''

    query_parameters = ''
    if pagination:
        query_parameters += urlencode(pagination)
    if query_parameters:
        query_parameters = '?' + query_parameters

    return _base_url() + route + resource_id_url + query_parameters