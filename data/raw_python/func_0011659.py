def update_subtask(client, subtask_id, revision, title=None, completed=None):
    '''
    Updates the subtask with the given ID

    See https://developer.wunderlist.com/documentation/endpoints/subtask for detailed parameter information
    '''
    if title is not None:
        _check_title_length(title, client.api)
    data = {
            'revision' : int(revision),
            'title' : title,
            'completed' : completed,
            }
    data = { key: value for key, value in data.items() if value is not None }
    endpoint = '/'.join([client.api.Endpoints.SUBTASKS, str(subtask_id)])
    response = client.authenticated_request(endpoint, 'PATCH', data=data)
    return response.json()