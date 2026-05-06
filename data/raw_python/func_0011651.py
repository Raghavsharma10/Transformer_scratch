def get_task_positions_objs(client, list_id):
    '''
    Gets a list containing the object that encapsulates information about the order lists are laid out in. This list will always contain exactly one object.

    See https://developer.wunderlist.com/documentation/endpoints/positions for more info

    Return:
    A list containing a single ListPositionsObj-mapped object
    '''
    params = {
            'list_id' : int(list_id)
            }
    response = client.authenticated_request(client.api.Endpoints.TASK_POSITIONS, params=params)
    return response.json()