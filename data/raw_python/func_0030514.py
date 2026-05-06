def get_build_configuration_by_name(name):
    """
    Returns the build configuration matching the name
    :param name: name of build configuration
    :return: The matching build configuration, or None if no match found
    """
    response = utils.checked_api_call(pnc_api.build_configs, 'get_all', q='name==' + name).content
    if not response:
        return None
    return response[0]