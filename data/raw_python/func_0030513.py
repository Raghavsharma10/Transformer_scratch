def get_build_configuration_id_by_name(name):
    """
    Returns the id of the build configuration matching name
    :param name: name of build configuration
    :return: id of the matching build configuration, or None if no match found
    """
    response = utils.checked_api_call(pnc_api.build_configs, 'get_all', q='name==' + name).content
    if not response:
        return None
    return response[0].id