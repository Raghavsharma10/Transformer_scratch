def id_exists(api, search_id):
    """
    Test if an ID exists within any arbitrary API
    :param api: api to search for search_id
    :param search_id: id to test for
    :return: True if an entity with ID search_id exists, false otherwise
    """
    response = utils.checked_api_call(api, 'get_specific', id=search_id)
    return response is not None