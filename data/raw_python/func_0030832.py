def get_id_by_name(api, search_name):
    """
    calls 'get_all' on the given API with a search name and returns the ID of the entity retrieved, if any, None otherwise
    :param api: api to search
    :param search_name: name to test for
    :return ID of entity matching search_name, None otherwise
    """
    entities = api.get_all(q='name==' + "'" + search_name + "'").content
    if entities:
        return entities[0].id
    return