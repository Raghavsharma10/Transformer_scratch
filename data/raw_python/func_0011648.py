def update_list(client, list_id, revision, title=None, public=None):
    '''
    Updates the list with the given ID to have the given properties

    See https://developer.wunderlist.com/documentation/endpoints/list for detailed parameter information
    '''
    if title is not None:
        _check_title_length(title, client.api)
    data = {
            'revision' : revision,
            'title' : title,
            'public' : public,
            }
    data = { key: value for key, value in data.items() if value is not None }
    endpoint = '/'.join([client.api.Endpoints.LISTS, str(list_id)])
    response = client.authenticated_request(endpoint, 'PATCH', data=data)
    return response.json()