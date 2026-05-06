def search(q, **kwargs):
    '''Returns a dictionary with the search results'''
    data = {'q': q}
    for key, value in kwargs.items():
        if value:
            if type(value) == bool:
                data[key] = 'on'
            else:
                data[key] = value
    return _unpaginated('search/?' + urlencode(data))