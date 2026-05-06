def config_id_exists(search_id):
    """
    Test if a build configuration matching search_id exists
    :param search_id: id to test for
    :return: True if a build configuration with search_id exists, False otherwise
    """
    response = utils.checked_api_call(pnc_api.build_configs, 'get_specific', id=search_id)
    if not response:
        return False
    return True