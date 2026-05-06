def get_verbatim(key, **kwargs):
    '''
    Gets a verbatim occurrence record without any interpretation

    :param key: [int] A GBIF occurrence key

    :return: A dictionary, of results

    Usage::

        from pygbif import occurrences
        occurrences.get_verbatim(key = 1258202889)
        occurrences.get_verbatim(key = 1227768771)
        occurrences.get_verbatim(key = 1227769518)
    '''
    url = gbif_baseurl + 'occurrence/' + str(key) + '/verbatim'
    out = gbif_GET(url, {}, **kwargs)
    return out