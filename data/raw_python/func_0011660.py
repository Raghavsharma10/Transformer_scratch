def delete_subtask(client, subtask_id, revision):
    ''' Deletes the subtask with the given ID provided the given revision equals the revision the server has '''
    params = {
            'revision' : int(revision),
            }
    endpoint = '/'.join([client.api.Endpoints.SUBTASKS, str(subtask_id)])
    client.authenticated_request(endpoint, 'DELETE', params=params)