def get_task_subtasks(client, task_id, completed=False):
    ''' Gets subtasks for task with given ID '''
    params = {
            'task_id' : int(task_id),
            'completed' : completed,
            }
    response = client.authenticated_request(client.api.Endpoints.SUBTASKS, params=params)
    return response.json()