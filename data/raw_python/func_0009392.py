def normalize_response_value(rv):
    """ Normalize the response value into a 3-tuple (rv, status, headers)
        :type rv: tuple|*
        :returns: tuple(rv, status, headers)
        :rtype: tuple(Response|JsonResponse|*, int|None, dict|None)
    """
    status = headers = None
    if isinstance(rv, tuple):
        rv, status, headers = rv + (None,) * (3 - len(rv))
    return rv, status, headers