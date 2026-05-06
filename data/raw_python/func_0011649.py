def get_list_positions_obj(client, positions_obj_id):
    '''
    Gets the object that defines how lists are ordered (there will always be only one of these)

    See https://developer.wunderlist.com/documentation/endpoints/positions for more info

    Return:
    A ListPositionsObj-mapped object defining the order of list layout
    '''
    return endpoint_helpers.get_endpoint_obj(client, client.api.Endpoints.LIST_POSITIONS, positions_obj_id)