def get_supported_resources(netid):
    """
    Returns list of Supported resources
    """
    url = _netid_supported_url(netid)
    response = get_resource(url)
    return _json_to_supported(response)