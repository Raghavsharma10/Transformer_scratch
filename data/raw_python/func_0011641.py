def get_task(client, task_id):
    ''' Gets task information for the given ID '''
    endpoint = '/'.join([client.api.Endpoints.TASKS, str(task_id)])
    response = client.authenticated_request(endpoint)
    return response.json()