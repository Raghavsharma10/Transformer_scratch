def decode_values(fct):
    ''' Decode base64 encoded responses from Consul storage '''
    def inner(*args, **kwargs):
        ''' decorator '''
        data = fct(*args, **kwargs)
        if 'error' not in data:
            for result in data:
                result['Value'] = base64.b64decode(result['Value'])
        return data
    return inner