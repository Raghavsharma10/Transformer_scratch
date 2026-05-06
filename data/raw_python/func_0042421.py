def humanize_api_path(api_path):
    """
    Converts an API path to a humaized string, for example:

        # >>> In [2]: humanize_api_path('/api/vlan/{id}')
        # >>> Out[2]: u'ApiVlanId'


    Parameters
    ----------
    api_path : str
        An API path string.

    Returns
    -------
    str - humazined form.
    """
    return reduce(lambda val, func: func(val),
                  [parameterize, underscore, camelize],
                  unicode(api_path))