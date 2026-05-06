def get_netid_categories(netid, category_codes):
    """
    Return a list of uwnetid.models Category objects
    corresponding to the netid and category code or list provided
    """
    url = _netid_category_url(netid, category_codes)
    response = get_resource(url)
    return _json_to_categories(response)