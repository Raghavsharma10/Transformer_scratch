def create_list(client, title):
    ''' Creates a new list with the given title '''
    _check_title_length(title, client.api)
    data = {
            'title' : title,
            }
    response = client.authenticated_request(client.api.Endpoints.LISTS, method='POST', data=data)
    return response.json()