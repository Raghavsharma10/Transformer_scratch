def make_json_response(rv):
    """ Make JsonResponse
    :param rv: Response: the object to encode, or tuple (response, status, headers)
    :type rv: tuple|*
    :rtype: JsonResponse
    """
    # Tuple of (response, status, headers)
    rv, status, headers = normalize_response_value(rv)

    # JsonResponse
    if isinstance(rv, JsonResponse):
        return rv

    # Data
    return JsonResponse(rv, status, headers)