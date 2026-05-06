def create_subtask(client, task_id, title, completed=False):
    ''' Creates a subtask with the given title under the task with the given ID '''
    _check_title_length(title, client.api)
    data = {
            'task_id' : int(task_id) if task_id else None,
            'title' : title,
            'completed' : completed,
            }
    data = { key: value for key, value in data.items() if value is not None }
    response = client.authenticated_request(client.api.Endpoints.SUBTASKS, 'POST', data=data)
    return response.json()