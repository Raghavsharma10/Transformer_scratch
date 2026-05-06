def get_entity(api, entity_id):
    """
    Generic "getSpecific" call that calls get_specific with the given id
    :param api: api to call get_specific on
    :param id: id of the entity to retrieve
    :return: REST entity
    """
    response = utils.checked_api_call(api, 'get_specific', id=entity_id)
    if response:
        return response.content
    return