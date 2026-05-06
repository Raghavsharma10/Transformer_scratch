def get_endpoint_obj(client, endpoint, object_id):
    ''' Tiny helper function that gets used all over the place to join the object ID to the endpoint and run a GET request, returning the result '''
    endpoint = '/'.join([endpoint, str(object_id)])
    return client.authenticated_request(endpoint).json()