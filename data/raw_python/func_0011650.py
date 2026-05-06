def update_list_positions_obj(client, positions_obj_id, revision, values):
    '''
    Updates the ordering of lists to have the given value. The given ID and revision should match the singleton object defining how lists are laid out.

    See https://developer.wunderlist.com/documentation/endpoints/positions for more info

    Return:
    The updated ListPositionsObj-mapped object defining the order of list layout
    '''
    return _update_positions_obj(client, client.api.Endpoints.LIST_POSITIONS, positions_obj_id, revision, values)