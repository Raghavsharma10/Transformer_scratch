def get_tasks(client, list_id, completed=False):
    ''' Gets un/completed tasks for the given list ID '''
    params = { 
            'list_id' : str(list_id), 
            'completed' : completed 
            }
    response = client.authenticated_request(client.api.Endpoints.TASKS, params=params)
    return response.json()