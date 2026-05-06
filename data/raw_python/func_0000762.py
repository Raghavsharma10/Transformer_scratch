def get_route_name(resource_uri):
    """ Get route name from RAML resource URI.

    :param resource_uri: String representing RAML resource URI.
    :returns string: String with route name, which is :resource_uri:
        stripped of non-word characters.
    """
    resource_uri = resource_uri.strip('/')
    resource_uri = re.sub('\W', '', resource_uri)
    return resource_uri