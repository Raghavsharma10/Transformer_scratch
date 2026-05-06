def create_task(client, list_id, title, assignee_id=None, completed=None, recurrence_type=None, recurrence_count=None, due_date=None, starred=None):
    ''' 
    Creates a task in the given list 

    See https://developer.wunderlist.com/documentation/endpoints/task for detailed parameter information
    '''
    _check_title_length(title, client.api)
    if (recurrence_type is None and recurrence_count is not None) or (recurrence_type is not None and recurrence_count is None):
        raise ValueError("recurrence_type and recurrence_count are required are required together")
    if due_date is not None:
        _check_date_format(due_date, client.api)
    data = {
            'list_id' : int(list_id) if list_id else None,
            'title' : title,
            'assignee_id' : int(assignee_id) if assignee_id else None,
            'completed' : completed,
            'recurrence_type' : recurrence_type,
            'recurrence_count' : int(recurrence_count) if recurrence_count else None,
            'due_date' : due_date,
            'starred' : starred,
            }
    data = { key: value for key, value in data.items() if value is not None }
    response = client.authenticated_request(client.api.Endpoints.TASKS, 'POST', data=data)
    return response.json()