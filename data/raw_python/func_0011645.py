def get_lists(client):
    ''' Gets all the client's lists '''
    response = client.authenticated_request(client.api.Endpoints.LISTS)
    return response.json()