def list_parameter_splitting(data, key, size_limit=8000, method='GET'):
    """
    Helper function split list used as input parameter for requests,
    since Apache has a limitation to 8190 Bytes for the lenght of an URI.
    We extended it to also split lfn and dataset list length for POST calls to avoid
    DB abuse even if there is no limit on hoe long the list can be. YG 2015-5-13
    :param data: url parameters
    :type data: dict
    :param key: key of parameter dictionary to split by lenght
    :type used_size: str
    :param size_limit: Split list in chunks of maximal size_limit bytes
    :type size_limit: int

    """
    values = list(data[key])
    data[key] = []

    for element in values:
        data[key].append(element)
        if method =='GET':
            size = len(urllib.urlencode(data))
        else:
            size = len(data)
        if size > size_limit:
            last_element = data[key].pop()
            yield data
            data[key] = [last_element]

    yield data