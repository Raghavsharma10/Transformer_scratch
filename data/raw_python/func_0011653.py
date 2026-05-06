def get_list_subtask_positions_objs(client, list_id):
    '''
    Gets all subtask positions objects for the tasks within a given list. This is a convenience method so you don't have to get all the list's tasks before getting subtasks, though I can't fathom how mass subtask reordering is useful.

    Returns:
    List of SubtaskPositionsObj-mapped objects representing the order of subtasks for the tasks within the given list
    '''
    params = {
            'list_id' : int(list_id)
            }
    response = client.authenticated_request(client.api.Endpoints.SUBTASK_POSITIONS, params=params)
    return response.json()