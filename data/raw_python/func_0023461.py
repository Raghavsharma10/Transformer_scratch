def json_wrap(function, *args, **kwargs):
    '''Return the json content of a function that returns a request'''
    try:
        # Some responses have data = None, but they generally signal a
        # successful API call as well.
        response = json.loads(function(*args, **kwargs).content)
        if 'data' in response:
            return response['data'] or True
        else:
            return response
    except Exception as exc:
        raise ClientException(exc)