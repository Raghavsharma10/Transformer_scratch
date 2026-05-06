def get_authorize_callback(endpoint, provider_id):
    """Get a qualified URL for the provider to return to upon authorization

    param: endpoint: Absolute path to append to the application's host
    """
    endpoint_prefix = config_value('BLUEPRINT_NAME')
    url = url_for(endpoint_prefix + '.' + endpoint, provider_id=provider_id)
    return request.url_root[:-1] + url