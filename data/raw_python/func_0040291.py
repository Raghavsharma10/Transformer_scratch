def json_response(data, status=200, serializer=None):
    """
    Returns an HttpResponse object containing JSON serialized data.

    The mime-type is set to application/json, and the charset to UTF-8.
    """
    return HttpResponse(json.dumps(data, default=serializer),
                        status=status,
                        content_type='application/json; charset=UTF-8')