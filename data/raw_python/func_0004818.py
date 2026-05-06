def get(key, **kwargs):
    '''
    Gets details for a single, interpreted occurrence

    :param key: [int] A GBIF occurrence key

    :return: A dictionary, of results

    Usage::

        from pygbif import occurrences
        occurrences.get(key = 1258202889)
        occurrences.get(key = 1227768771)
        occurrences.get(key = 1227769518)
    '''
    url = gbif_baseurl + 'occurrence/' + str(key)
    out = gbif_GET(url, {}, **kwargs)
    return out