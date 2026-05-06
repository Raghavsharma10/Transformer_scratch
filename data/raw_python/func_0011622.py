def update_endpoint_obj(client, endpoint, object_id, revision, data):
    ''' 
    Helper method to ease the repetitiveness of updating an... SO VERY DRY 
    
    (That's a doubly-effective pun becuase my predecessor - https://github.com/bsmt/wunderpy - found maintaing a Python Wunderlist API to be "as tedious and boring as a liberal arts school poetry slam") 
    '''
    data['revision'] = int(revision)
    endpoint = '/'.join([endpoint, str(object_id)])
    return client.authenticated_request(endpoint, 'PATCH', data=data).json()