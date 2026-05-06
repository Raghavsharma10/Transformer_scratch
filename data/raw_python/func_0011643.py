def update_task(client, task_id, revision, title=None, assignee_id=None, completed=None, recurrence_type=None, recurrence_count=None, due_date=None, starred=None, remove=None):
    '''
    Updates the task with the given ID

    See https://developer.wunderlist.com/documentation/endpoints/task for detailed parameter information
    '''
    if title is not None:
        _check_title_length(title, client.api)
    if (recurrence_type is None and recurrence_count is not None) or (recurrence_type is not None and recurrence_count is None):
        raise ValueError("recurrence_type and recurrence_count are required are required together")
    if due_date is not None:
        _check_date_format(due_date, client.api)
    data = {
            'revision' : int(revision),
            'title' : title,
            'assignee_id' : int(assignee_id) if assignee_id else None,
            'completed' : completed,
            'recurrence_type' : recurrence_type,
            'recurrence_count' : int(recurrence_count) if recurrence_count else None,
            'due_date' : due_date,
            'starred' : starred,
            'remove' : remove,
            }
    data = { key: value for key, value in data.items() if value is not None }
    endpoint = '/'.join([client.api.Endpoints.TASKS, str(task_id)])
    response = client.authenticated_request(endpoint, 'PATCH', data=data)
    return response.json()