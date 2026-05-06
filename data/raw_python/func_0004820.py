def get_fragment(key, **kwargs):
    '''
    Get a single occurrence fragment in its raw form (xml or json)

    :param key: [int] A GBIF occurrence key

    :return: A dictionary, of results

    Usage::

        from pygbif import occurrences
        occurrences.get_fragment(key = 1052909293)
        occurrences.get_fragment(key = 1227768771)
        occurrences.get_fragment(key = 1227769518)
    '''
    url = gbif_baseurl + 'occurrence/' + str(key) + '/fragment'
    out = gbif_GET(url, {}, **kwargs)
    return out