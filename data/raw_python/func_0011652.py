def get_task_subtask_positions_objs(client, task_id):
    '''
    Gets a list of the positions of a single task's subtasks

    Each task should (will?) only have one positions object defining how its subtasks are laid out
    '''
    params = {
            'task_id' : int(task_id)
            }
    response = client.authenticated_request(client.api.Endpoints.SUBTASK_POSITIONS, params=params)
    return response.json()