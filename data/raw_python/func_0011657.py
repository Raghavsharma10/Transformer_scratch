def get_subtask(client, subtask_id):
    ''' Gets the subtask with the given ID '''
    endpoint = '/'.join([client.api.Endpoints.SUBTASKS, str(subtask_id)])
    response = client.authenticated_request(endpoint)
    return response.json()