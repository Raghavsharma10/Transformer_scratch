def get_list_subtasks(client, list_id, completed=False):
    ''' Gets subtasks for the list with given ID '''
    params = {
            'list_id' : int(list_id),
            'completed' : completed,
            }
    response = client.authenticated_request(client.api.Endpoints.SUBTASKS, params=params)
    return response.json()