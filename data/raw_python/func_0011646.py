def get_list(client, list_id):
    ''' Gets the given list '''
    endpoint = '/'.join([client.api.Endpoints.LISTS, str(list_id)])
    response = client.authenticated_request(endpoint)
    return response.json()